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
from torch.func import grad, hessian

DTYPE = torch.float64


def _cone_volume(xe):
    return torch.dot(xe[0], torch.linalg.cross(xe[1], xe[2])) / 6.0


# Cone volumes and their derivatives are written out by hand (torch.func's vmap/grad/hessian of _cone_volume give
# the same numbers, but its overhead was ~20% of a Newton evaluation). V = x0 . (x1 x x2) / 6 per face (E, 3, 3).

def cone_volume(xe):
    return (xe[:, 0] * torch.linalg.cross(xe[:, 1], xe[:, 2])).sum(-1) / 6.0


def cone_volume_grad(xe):
    """dV/dx (E, 3 nodes, 3): dV/dx0 = x1 x x2 / 6 and cyclic."""
    x0, x1, x2 = xe[:, 0], xe[:, 1], xe[:, 2]
    return torch.stack((torch.linalg.cross(x1, x2), torch.linalg.cross(x2, x0), torch.linalg.cross(x0, x1)), 1) / 6.0


def _levi_civita(v):
    """eps_ijk v_k (E, 3, 3)."""
    z = torch.zeros_like(v[:, 0])
    return torch.stack((torch.stack((z, v[:, 2], -v[:, 1]), -1),
                        torch.stack((-v[:, 2], z, v[:, 0]), -1),
                        torch.stack((v[:, 1], -v[:, 0], z), -1)), 1)


def cone_volume_hess(xe):
    """d2V/dx2 (E, 3, 3, 3, 3) indexed [node p, i, node q, j]: the (p, p+1) block is eps_ijk x_{p+2,k} / 6, the
    (p+1, p) block its transpose (= minus it), diagonal blocks zero."""
    H = xe.new_zeros(xe.shape[0], 3, 3, 3, 3)
    for p in range(3):
        q, r = (p + 1) % 3, (p + 2) % 3
        B = _levi_civita(xe[:, r]) / 6.0
        H[:, p, :, q, :] = B
        H[:, q, :, p, :] = -B
    return H

_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(16)


def shell_cone_volume(shell, x) -> torch.Tensor:
    return cone_volume(x[shell.faces] - shell.volume_origin).sum()


def wall_volume(body, rest: bool = False) -> float:
    """Volume measure of a chamber wall (its changes are what count): the cone volume of a shell, or the
    swept volume of a wall with its own volume model (e.g. an EmpiricalMembrane, 0 at rest)."""
    if hasattr(body, "swept_volume"):
        return 0.0 if rest else body.swept_volume()
    return shell_cone_volume(body, body.X if rest else body.x).item()


def wall_volume_terms(body, side: int, tangent: bool):
    """(local dofs (E, d), d(side V)/du (E, d), d2(side V)/du2 (E, d, d) or None) of one wall."""
    if hasattr(body, "volume_terms"):
        return body.volume_terms(side, tangent)
    xe = body.x[body.faces] - body.volume_origin
    return (body.face_dofs, side * cone_volume_grad(xe).reshape(-1, 9),
            side * cone_volume_hess(xe).reshape(-1, 9, 9) if tangent else None)


def _cap_volume(ring):
    """Cone volume (apex at the origin) of the fan closing a boundary loop: triangles
    (ring[k+1], ring[k], centroid), i.e. the loop's half-edges reversed."""
    c = ring.mean(dim=0)
    nxt = torch.roll(ring, -1, dims=0)
    return (nxt * torch.linalg.cross(ring, c.expand_as(ring))).sum() / 6.0


_cap_grad, _cap_hess = grad(_cap_volume), hessian(_cap_volume)


class SurfacePatch:
    """
    Part of a body's boundary that walls a chamber (e.g. the inside of a tube, split into an
    input and an output half). Unlike a clamped membrane, a patch can have boundary loops that
    move (where the two halves of a tube meet); every loop with a free node is closed by a fan
    of triangles to the loop's centroid, so the enclosed volume stays exact and independent of
    the cone apex. The cap's share of the pressure goes to the loop's nodes: it stands for the
    axial force that the pressure drop across a constriction puts on the walls (viscous shear in
    steady flow), and cancels where both sides are at the same pressure.
    """

    def __init__(self, body, face_indices, side: int):
        self.body, self.side = body, side
        F = body.faces_np[np.asarray(face_indices, dtype=np.int64)]
        self.faces = torch.as_tensor(F, device=body.device)
        comp = torch.arange(3, device=body.device)
        self.face_dofs = (3 * self.faces[:, :, None] + comp).reshape(-1, 9)
        self.origin = body.X[torch.as_tensor(np.unique(F), device=body.device)].mean(dim=0)
        fixed = body.fixed.cpu().numpy()
        self.loops = [torch.as_tensor(loop, device=body.device)
                      for loop in boundary_loops(F) if not fixed[loop].all()]
        self.loop_dofs = [(3 * loop[:, None] + comp).reshape(-1) for loop in self.loops]
        self.rest = self.volume(body.X)

    def volume(self, x) -> float:
        v = cone_volume(x[self.faces] - self.origin).sum()
        for loop in self.loops:
            v = v + _cap_volume(x[loop] - self.origin)
        return float(v)

    def terms(self, tangent: bool):
        """[(body, dofs (E, d), gradient (E, d), Hessian (E, d, d) or None)] of side * volume."""
        x = self.body.x
        xe = x[self.faces] - self.origin
        out = [(self.body, self.face_dofs, self.side * cone_volume_grad(xe).reshape(-1, 9),
                self.side * cone_volume_hess(xe).reshape(-1, 9, 9) if tangent else None)]
        for loop, dofs in zip(self.loops, self.loop_dofs):
            ring = x[loop] - self.origin
            g = self.side * _cap_grad(ring).reshape(1, -1)
            H = self.side * _cap_hess(ring).reshape(1, len(dofs), len(dofs)) if tangent else None
            out.append((self.body, dofs[None], g, H))
        return out


def boundary_loops(F):
    """Boundary loops of an oriented triangle patch as node lists, following the half-edges that
    have no twin (in the patch's orientation)."""
    half = {}
    for a, b, c in F:
        for i, j in ((a, b), (b, c), (c, a)):
            half[(int(i), int(j))] = True
    nxt = {}
    for i, j in half:
        if (j, i) not in half:
            nxt[i] = j
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, node = [], start
        while node not in seen and node in nxt:
            seen.add(node)
            loop.append(node)
            node = nxt[node]
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
    return loops


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
        self.patches = []     # SurfacePatch walls of solid bodies
        self.delta_volume = 0.0
        self.P = self.pressure(0.0)
        self.pressure_hist = []
        self.volume_hist = []

    def add_boundary(self, shell, side: int):
        if side not in (1, -1):
            raise ValueError("side must be +1 (normal points out of the volume) or -1.")
        rest = wall_volume(shell, rest=True)
        self.boundaries.append((shell, side, rest))
        shell.fluid_volumes.append(self)

    def add_patch(self, body, face_indices, side: int) -> "SurfacePatch":
        """Part of a solid's boundary walls this chamber. side = +1 when the faces' normals point
        out of the chamber (a solid's outward normals point *into* a cavity it surrounds: -1)."""
        if side not in (1, -1):
            raise ValueError("side must be +1 (normal points out of the volume) or -1.")
        patch = SurfacePatch(body, face_indices, side)
        self.patches.append(patch)
        body.fluid_volumes.append(self)
        return patch

    # -----------------------------
    # Geometry
    # -----------------------------

    def compute_delta_volume(self) -> float:
        return (sum(side * (wall_volume(shell) - rest) for shell, side, rest in self.boundaries)
                + sum(p.side * (p.volume(p.body.x) - p.rest) for p in self.patches))

    def volume_terms(self, tangent: bool):
        """[(body, local dofs (E, d), d(dV)/du (E, d), d2(dV)/du2 (E, d, d) or None)] over all walls."""
        out = []
        for shell, side, _ in self.boundaries:
            out.append((shell, *wall_volume_terms(shell, side, tangent)))
        for patch in self.patches:
            out.extend(patch.terms(tangent))
        return out

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
