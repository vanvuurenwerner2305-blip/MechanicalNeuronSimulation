"""
Moving rigid bodies and the couplings that tie them to the deformable parts.

RigidBody: 6 dofs q = [t, theta], translation t and rotation vector theta about the rest
centroid c0, so a body point X moves to  x = c0 + t + R(theta) (X - c0)  (Rodrigues).
The body itself stores no energy; it is moved only by its couplings:

  RigidTie      - nodes of a membrane/shell glued to the body (e.g. a pusher bonded under a
                  membrane): penalty 1/2 k A_i |x_i - x_body(X_i)|^2.
  MovingContact - surface nodes of a deformable part against the body's closed surface (e.g. the
                  pusher pressing on a tube): penalty 1/2 k A_i min(sd - offset, 0)^2 with the
                  signed distance evaluated in the body frame,  sd(x) = sd_0(R^T (x - c0 - t) + c0).

Couplings give the solver (global dofs, energy, gradient, Hessian) per node, like the contact
terms; the Hessians are exact (torch.func). For MovingContact the obstacle's signed distance is
replaced by its second-order expansion around the current point, which has the same value,
gradient and Hessian there.
"""
import math

import numpy as np
import torch
from torch.func import grad, hessian, vmap

from .contact import CachedSignedDistance, Obstacle, ObstacleField

DTYPE = torch.float64


def rotation_matrix(theta):
    """Rodrigues' formula, smooth (with its derivatives) through theta = 0."""
    s2 = (theta * theta).sum()
    small = s2 < 1e-8
    safe = torch.where(small, torch.ones_like(s2), s2)
    s = torch.sqrt(safe)
    a = torch.where(small, 1.0 - s2 / 6.0 + s2 ** 2 / 120.0, torch.sin(s) / s)
    b = torch.where(small, 0.5 - s2 / 24.0 + s2 ** 2 / 720.0, (1.0 - torch.cos(s)) / safe)
    zero = torch.zeros_like(s2)
    K = torch.stack([torch.stack([zero, -theta[2], theta[1]]),
                     torch.stack([theta[2], zero, -theta[0]]),
                     torch.stack([-theta[1], theta[0], zero])])
    return torch.eye(3, dtype=theta.dtype, device=theta.device) + a * K + b * (K @ K)


class RigidBody:
    sheet_contact = False
    is_rigid = True
    n_nodes = 0          # no node dofs (the solver's per-node bookkeeping skips it)
    n_edges = 0
    n_dof = 6
    thickness = 0.0
    youngs_modulus = 0.0
    material = "rigid"

    def __init__(self, vertices, faces, color: str = "#8d99ae", name: str = None, device="cpu"):
        self.device = torch.device(device)
        self.color, self.name = color, name
        self.X = torch.as_tensor(np.asarray(vertices, float), dtype=DTYPE, device=self.device)
        self.faces = torch.as_tensor(np.asarray(faces, np.int64), device=self.device)
        self.faces_np = np.asarray(faces, np.int64)
        self.center = self.X.mean(dim=0)
        self.q = torch.zeros(6, dtype=DTYPE, device=self.device)
        self.fixed = torch.zeros(0, dtype=torch.bool, device=self.device)
        self.fixed_dofs = torch.zeros(6, dtype=torch.bool, device=self.device)
        self.coord_mask = np.array([True, True, True, False, False, False])  # translations are lengths
        self.fluid_volumes = []
        self.obstacle = Obstacle(vertices, faces, device=self.device)  # rest pose, for body-frame queries

    def get_state(self):
        return self.q

    def set_state(self, u):
        self.q = u

    def reset(self):
        self.q = torch.zeros_like(self.q)

    def element_terms(self, tangent=True):
        return iter(())

    def transform(self, points, q=None):
        q = self.q if q is None else q
        R = rotation_matrix(q[3:])
        return self.center + q[:3] + (points - self.center) @ R.T

    def to_body(self, points, q=None):
        q = self.q if q is None else q
        R = rotation_matrix(q[3:])
        return self.center + (points - self.center - q[:3]) @ R

    @property
    def x(self):
        """Current surface vertices (for display and the load-step history)."""
        return self.transform(self.X)

    @property
    def translation(self):
        return self.q[:3]

    @property
    def rotation(self):
        return self.q[3:]


# -----------------------------
# Couplings
# -----------------------------

def _tie_energy(z, X, c, kA):
    """z = [node x (3), body t (3), theta (3)]."""
    x, t, theta = z[:3], z[3:6], z[6:9]
    target = c + t + rotation_matrix(theta) @ (X - c)
    d = x - target
    return 0.5 * kA * (d * d).sum()


_tie_e, _tie_g, _tie_h = vmap(_tie_energy), vmap(grad(_tie_energy)), vmap(hessian(_tie_energy))


class RigidTie:
    """Glue nodes of a deformable body (membrane) to a RigidBody with a stiff penalty."""

    def __init__(self, body, nodes, rigid: RigidBody, stiffness: float):
        self.body, self.rigid, self.stiffness = body, rigid, float(stiffness)
        self.nodes = torch.as_tensor(np.asarray(nodes, np.int64), device=body.device)
        self.kA = self.stiffness * body.nodal_area[self.nodes]

    def bind(self, solver):
        self.solver = solver

    def terms(self, tangent=True):
        if not len(self.nodes):
            return []
        z = torch.cat([self.body.x[self.nodes], self.rigid.q.expand(len(self.nodes), 6)], dim=1)
        args = (z, self.body.X[self.nodes], self.rigid.center.expand(len(self.nodes), 3), self.kA)
        dofs = _coupled_dofs(self.solver, self.body, self.nodes, self.rigid)
        return [(dofs, _tie_e(*args), _tie_g(*args), _tie_h(*args) if tangent else None)]

    def set_stiffness(self, k):
        self.stiffness = float(k)
        self.kA = self.stiffness * self.body.nodal_area[self.nodes]


def _surrogate_sd(z, c, sd0, n0, H0, y0):
    """Second-order expansion of the body-frame signed distance around y0 (value sd0, gradient n0,
    Hessian H0), as a function of z = [node x, body t, theta]."""
    x, t, theta = z[:3], z[3:6], z[6:9]
    y = c + rotation_matrix(theta).T @ (x - c - t)
    dy = y - y0
    return sd0 + n0 @ dy + 0.5 * dy @ H0 @ dy


def _contact_energy(z, c, sd0, n0, H0, y0, offset, kA):
    gap = _surrogate_sd(z, c, sd0, n0, H0, y0) - offset
    return 0.5 * kA * gap ** 2


_mc_g = vmap(grad(_contact_energy))
_mc_h = vmap(hessian(_contact_energy))


class MovingContact:
    """Surface nodes of a deformable body against a moving RigidBody (one-sided penalty)."""

    def __init__(self, body, rigid: RigidBody, stiffness: float, nodes=None, max_reach: float = None):
        self.body, self.rigid, self.stiffness = body, rigid, float(stiffness)
        if nodes is None:
            nodes = body.surface_nodes if hasattr(body, "surface_nodes") else torch.arange(body.n_nodes)
        nodes = torch.as_tensor(nodes, device=body.device)
        nodes = nodes[~body.fixed[nodes]]
        self.field = ObstacleField([rigid.obstacle])
        size = (rigid.X.max(0).values - rigid.X.min(0).values).norm().item()
        reach = max_reach if max_reach is not None else 0.05 * size
        rest_sd, _ = self.field.signed_distance(body.X[nodes], max_distance=reach)
        # Rest geometry that already touches (or, faceted, slightly overlaps) the body is neutral
        self.offset = torch.clamp(rest_sd, max=0.0)
        self.nodes = nodes
        self.cache = CachedSignedDistance(self.field, max(reach, 1e-9), skin=max(reach, 0.01 * size))

    def bind(self, solver):
        self.solver = solver

    def set_stiffness(self, k):
        self.stiffness = float(k)

    def terms(self, tangent=True):
        if not len(self.nodes):
            return []
        x = self.body.x[self.nodes]
        y = self.rigid.to_body(x)
        sd, n0, H0 = self.cache.signed_distance(y, hessian=True)
        gap = sd - self.offset
        active = gap < 0
        if not active.any():
            return []
        nodes = self.nodes[active]
        m = len(nodes)
        kA = self.stiffness * self.body.nodal_area[nodes]
        z = torch.cat([x[active], self.rigid.q.expand(m, 6)], dim=1)
        args = (z, self.rigid.center.expand(m, 3), sd[active], n0[active], H0[active], y[active],
                self.offset[active], kA)
        energy = 0.5 * kA * gap[active] ** 2
        dofs = _coupled_dofs(self.solver, self.body, nodes, self.rigid)
        return [(dofs, energy, _mc_g(*args), _mc_h(*args) if tangent else None)]


def _coupled_dofs(solver, body, nodes, rigid):
    arange = torch.arange(3, device=body.device)
    node_dofs = solver.dof_offset[id(body)] + 3 * nodes[:, None] + arange
    rigid_dofs = (solver.dof_offset[id(rigid)] + torch.arange(6, device=body.device)).expand(len(nodes), 6)
    return torch.cat([node_dofs, rigid_dofs], dim=1)
