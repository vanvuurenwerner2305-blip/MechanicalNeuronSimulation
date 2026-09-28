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
                 color: str = "blue",
                 name: str = None):
        """
        P0             : pressure at the rest volume.
        bulk_stiffness : K in P = P0 - K * dV (dP/dV). K = 0 gives a constant pressure reservoir.
        pressure_law   : optional callable dV (torch scalar) -> P (torch scalar), replacing the
                         linear law, e.g. an ideal gas lambda dV: P0 * V0 / (V0 + dV).
                         Must be differentiable with torch.
        initial_volume : optional rest volume, only used for reporting `volume`.
        """
        self.P0 = P0
        self.bulk_stiffness = bulk_stiffness
        self.pressure_law = pressure_law
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
    def volume(self):
        return None if self.initial_volume is None else self.initial_volume + self.delta_volume

    # -----------------------------
    # Pressure law
    # -----------------------------

    def pressure(self, dV: float) -> float:
        if self.pressure_law is None:
            return self.P0 - self.bulk_stiffness * dV
        return float(self.pressure_law(torch.tensor(dV, dtype=DTYPE)))

    def pressure_slope(self, dV: float) -> float:
        if self.pressure_law is None:
            return -self.bulk_stiffness
        t = torch.tensor(dV, dtype=DTYPE, requires_grad=True)
        (slope,) = torch.autograd.grad(self.pressure_law(t), t)
        return float(slope)

    def pressure_potential(self, dV: float) -> float:
        """Integral of P from 0 to dV (the work done by the fluid)."""
        if self.pressure_law is None:
            return self.P0 * dV - 0.5 * self.bulk_stiffness * dV ** 2
        s = 0.5 * dV * (_GAUSS_X + 1.0)
        return 0.5 * dV * sum(w * self.pressure(si) for si, w in zip(s, _GAUSS_W))

    # -----------------------------
    # State
    # -----------------------------

    def update(self, load_factor: float = 1.0):
        self.delta_volume = self.compute_delta_volume()
        self.P = load_factor * self.pressure(self.delta_volume)

    def reset(self):
        self.delta_volume = 0.0
        self.P = self.pressure(0.0)
        self.pressure_hist = []
        self.volume_hist = []
