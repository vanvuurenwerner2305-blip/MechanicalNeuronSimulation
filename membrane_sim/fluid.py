"""
Fluid (or gas) chambers bounded by shells.

Only the *change* of a chamber's volume is needed, and it can be computed from the
deforming shells alone: because shell boundary nodes are pinned, the deformed and the
rest shell surfaces together close a volume, so

    dV = sum_shells side * (V_cone(x) - V_cone(X)),   V_cone = 1/6 sum_faces x0 . (x1 x x2)

independent of the (rigid) walls and of the cone apex. side = +1 when the shell normal
points out of the chamber. Pressure is a function of dV, and the pressure load on the
shells is P * d(dV)/dx, which is exactly the (follower) pressure load and gives a
symmetric tangent: P * d2V/dx2 + P'(dV) * g g^T with g = d(dV)/dx.
"""
import inspect
import math

import numpy as np
import torch
from torch.func import vmap, grad, hessian

DTYPE = torch.float64


def _cone_volume(xe):
    return torch.dot(xe[0], torch.linalg.cross(xe[1], xe[2])) / 6.0


cone_volume = vmap(_cone_volume)
cone_volume_grad = vmap(grad(_cone_volume))
cone_volume_hess = vmap(hessian(_cone_volume))

_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(16)


def shell_cone_volume(shell, x) -> torch.Tensor:
    return cone_volume(x[shell.faces] - shell.volume_origin).sum()


class FluidVolume:
    def __init__(self,
                 P0: float = 0.0,
                 bulk_stiffness: float = 0.0,
                 pressure_law=None,
                 initial_volume: float = None,
                 gas_volume: float = None,
                 liquid_volume: float = None,
                 atmospheric_pressure: float = 0.101325,
                 color: str = "blue",
                 name: str = None):
        """
        P0             : gauge pressure at the rest volume (0 = atmospheric).
        bulk_stiffness : K in P = P0 - K * dV (dP/dV). K = 0 gives a constant pressure reservoir.
        pressure_law   : optional callable dV -> P or (dV, P0) -> P (torch scalars), replacing
                         the linear law, e.g. an ideal gas lambda dV, P0: P0 * V0 / (V0 + dV).
                         Must be differentiable with torch. Taking P0 as an argument lets P0 be
                         changed later (sweeps, warm starts).
        initial_volume : optional rest volume, only used for reporting `volume`.
        gas_volume     : makes the chamber a sealed ideal gas (isothermal, Boyle's law) with this
                         much gas at rest; the rest of the chamber is incompressible liquid, so the
                         whole volume change goes into the gas:
                             P = (P_atm + P0) * V_gas / (V_gas + dV) - P_atm   (gauge)
                         The gas can not be compressed to zero volume (the energy becomes infinite).
        liquid_volume  : with the linear law: the sealed chamber holds this much liquid (needs
                         initial_volume), so the liquid is at rest at this volume, not at the chamber's:
                             P = P0 - K * (initial_volume + dV - liquid_volume)
                         Less liquid than the chamber gives suction (negative pressure) that pulls the
                         walls in; more liquid inflates it.
        atmospheric_pressure : P_atm for the gas law, in the model's pressure units (default MPa).
        """
        self.P0 = P0
        self.bulk_stiffness = bulk_stiffness
        self.pressure_law = pressure_law
        self.gas_volume = gas_volume
        self.atmospheric_pressure = atmospheric_pressure
        if gas_volume is not None and gas_volume <= 0:
            raise ValueError("gas_volume must be positive.")
        self.excess_volume = 0.0  # chamber volume - liquid volume at rest (linear law)
        if liquid_volume is not None:
            if liquid_volume <= 0 or initial_volume is None:
                raise ValueError("liquid_volume must be positive and needs initial_volume.")
            self.excess_volume = initial_volume - liquid_volume
        self.liquid_volume = liquid_volume
        self._law_takes_P0 = pressure_law is not None and len(inspect.signature(pressure_law).parameters) >= 2
        self.start_P0 = None   # P0 the current load path starts from (None: from zero pressure)
        self.solved_P0 = None  # P0 of the last converged solve
        self.initial_volume = initial_volume
        self.color = color
        self.name = name

        self.boundaries = []  # (shell, side, rest cone volume)
        self.delta_volume = 0.0
        self.P = self.pressure(0.0)
        self.pressure_hist = []
        self.volume_hist = []

    def add_boundary(self, shell, side: int):
        if side not in (1, -1):
            raise ValueError("side must be +1 (normal points out of the volume) or -1.")
        rest = shell_cone_volume(shell, shell.X).item()
        self.boundaries.append((shell, side, rest))
        shell.fluid_volumes.append(self)

    # -----------------------------
    # Geometry
    # -----------------------------

    def compute_delta_volume(self) -> float:
        return sum(side * (shell_cone_volume(shell, shell.x).item() - rest)
                   for shell, side, rest in self.boundaries)

    @property
    def is_closed(self) -> bool:
        """Sealed chamber whose pressure depends on its volume (not an input reservoir or vent)."""
        return self.gas_volume is not None or self.pressure_law is not None or self.bulk_stiffness != 0

    @property
    def volume(self):
        return None if self.initial_volume is None else self.initial_volume + self.delta_volume

    # -----------------------------
    # Pressure law
    # -----------------------------

    def _law(self, dV, P0):
        if self._law_takes_P0:
            return self.pressure_law(dV, P0)
        return self.pressure_law(dV)

    def pressure(self, dV: float, P0: float = None) -> float:
        P0 = self.P0 if P0 is None else P0
        if self.gas_volume is not None:
            gas = self.gas_volume + dV
            if gas <= 0:
                return math.inf
            return (self.atmospheric_pressure + P0) * self.gas_volume / gas - self.atmospheric_pressure
        if self.pressure_law is None:
            return P0 - self.bulk_stiffness * (dV + self.excess_volume)
        return float(self._law(torch.tensor(dV, dtype=DTYPE), P0))

    def pressure_slope(self, dV: float, P0: float = None) -> float:
        P0 = self.P0 if P0 is None else P0
        if self.gas_volume is not None:
            gas = self.gas_volume + dV
            if gas <= 0:
                return -math.inf
            return -(self.atmospheric_pressure + P0) * self.gas_volume / gas ** 2
        if self.pressure_law is None:
            return -self.bulk_stiffness
        t = torch.tensor(dV, dtype=DTYPE, requires_grad=True)
        (slope,) = torch.autograd.grad(self._law(t, P0), t)
        return float(slope)

    def pressure_potential(self, dV: float, P0: float = None) -> float:
        """Integral of P from 0 to dV (the work done by the fluid)."""
        P0 = self.P0 if P0 is None else P0
        if self.gas_volume is not None:
            gas = self.gas_volume + dV
            if gas <= 0:
                return -math.inf  # total energy +inf: such a state is rejected by the solver
            return ((self.atmospheric_pressure + P0) * self.gas_volume * math.log(gas / self.gas_volume)
                    - self.atmospheric_pressure * dV)
        if self.pressure_law is None:
            return P0 * dV - self.bulk_stiffness * (self.excess_volume * dV + 0.5 * dV ** 2)
        s = 0.5 * dV * (_GAUSS_X + 1.0)
        return 0.5 * dV * sum(w * self.pressure(si, P0) for si, w in zip(s, _GAUSS_W))

    def load_state(self, dV: float, load_factor: float):
        """
        (P, dP/dV, potential) along the load path. A fresh solve ramps the pressure from zero:
        lambda * P(dV). A warm-started solve blends from the previously solved P0 to the new one:
        (1 - lambda) * P(dV; P0_start) + lambda * P(dV; P0).
        """
        lam = load_factor
        P, slope, work = lam * self.pressure(dV), lam * self.pressure_slope(dV), lam * self.pressure_potential(dV)
        if self.start_P0 is not None and lam < 1.0:
            P += (1 - lam) * self.pressure(dV, self.start_P0)
            slope += (1 - lam) * self.pressure_slope(dV, self.start_P0)
            work += (1 - lam) * self.pressure_potential(dV, self.start_P0)
        return P, slope, work

    # -----------------------------
    # State
    # -----------------------------

    def update(self, load_factor: float = 1.0):
        self.delta_volume = self.compute_delta_volume()
        self.P = self.load_state(self.delta_volume, load_factor)[0]

    def reset(self):
        self.delta_volume = 0.0
        self.start_P0 = None
        self.solved_P0 = None
        self.P = self.pressure(0.0)
        self.pressure_hist = []
        self.volume_hist = []
