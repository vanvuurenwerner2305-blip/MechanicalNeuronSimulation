"""
Couplings between a body and the surface of a deformable solid (membrane_sim.solid.Solid).

SolidContact - surface nodes of body A kept outside the closed surface of solid B (e.g. a soft
               pusher pressing on a soft tube). Each node is paired with its closest triangle of
               B; the signed distance to that triangle is its plane distance when the closest
               point lies inside the triangle and the distance to the closest edge or vertex,
               signed by the triangle's normal, otherwise. Penalty
                   E = 1/2 k A_node min(d - g, 0)^2,   g = min(0, rest distance),
               so bodies that touch (or, faceted, slightly overlap) at rest start neutral.
               Gradient and Hessian are exact over the node and the triangle's three vertices
               (torch.func); the closest triangle can change between evaluations, which makes
               the energy only C0 where a node crosses from one triangle to the next.
SurfaceTie   - nodes of a membrane bonded to a solid's surface: each node follows the material
               point of the solid it started on (barycentric on the closest triangle), with its
               rest offset: E = 1/2 k A_node |x - sum_k w_k x_k - r0|^2. The offset r0 does not
               rotate with the surface, which is exact for the small rotations of a bonded pad.

Both give the solver (global dofs, energy, gradient, Hessian), like the other couplings.
"""
import numpy as np
import torch
from torch.func import grad, hessian, vmap

from .contact import _closest_point_pairs
from .shell_contact import closest_on_mesh

DTYPE = torch.float64


def _signed_distance(z):
    """z = [node p (3), triangle a, b, c (9)]."""
    p, a, b, c = z[0:3], z[3:6], z[6:9], z[9:12]
    q, feature = _closest_point_pairs(p[None], a[None], b[None], c[None])
    n = torch.linalg.cross(b - a, c - a)
    n = n / torch.linalg.norm(n)
    plane = torch.dot(p - a, n)
    diff = p - q[0]
    # distance to an edge/vertex; the square root is only taken of a positive number
    d2 = torch.dot(diff, diff)
    dist = torch.sqrt(torch.where(d2 > 1e-30, d2, torch.ones_like(d2)))
    edge = torch.where(plane < 0, -dist, dist)
    return torch.where(feature[0] == 0, plane, edge)


def _contact_energy(z, g, kA):
    gap = _signed_distance(z) - g
    return 0.5 * kA * gap ** 2


_sd = vmap(_signed_distance)
_ce_g = vmap(grad(_contact_energy))
_ce_h = vmap(hessian(_contact_energy))


class SolidContact:
    def __init__(self, A, B, stiffness: float, reach: float, nodes=None):
        """
        A     : body whose surface nodes are kept out (Solid or Shell).
        B     : Solid whose closed boundary surface they are kept out of.
        reach : search distance for the closest triangle (larger than the deepest penetration).
        """
        self.A, self.B, self.stiffness, self.reach = A, B, float(stiffness), float(reach)
        if nodes is None:
            nodes = A.surface_nodes if hasattr(A, "surface_nodes") else torch.arange(A.n_nodes)
        nodes = torch.as_tensor(nodes, device=A.device)
        nodes = nodes[~A.fixed[nodes]]
        # only nodes near B at rest can meet it without passing far through other parts first
        lo, hi = B.X.min(0).values - 2 * reach, B.X.max(0).values + 2 * reach
        near = ((A.X[nodes] >= lo) & (A.X[nodes] <= hi)).all(dim=1)
        self.nodes = nodes[near]
        self.offset = torch.zeros(len(self.nodes), dtype=DTYPE, device=A.device)
        if len(self.nodes):
            sd, _ = self._query(A.X[self.nodes], B.X)
            self.offset = torch.clamp(torch.nan_to_num(sd, nan=0.0, posinf=0.0), max=0.0)

    def bind(self, solver):
        self.solver = solver

    def set_stiffness(self, k):
        self.stiffness = float(k)

    def _query(self, points, xB):
        dist, _, tri = closest_on_mesh(points, xB, self.B.faces, self.reach)
        found = torch.isfinite(dist)
        sd = torch.full_like(dist, float("inf"))
        if found.any():
            z = torch.cat([points[found], xB[self.B.faces[tri[found]]].reshape(-1, 9)], dim=1)
            sd[found] = _sd(z)
        return sd, tri

    def terms(self, tangent=True):
        if not len(self.nodes):
            return []
        points = self.A.x[self.nodes]
        sd, tri = self._query(points, self.B.x)
        active = sd < self.offset
        if not active.any():
            return []
        nodes, tri = self.nodes[active], tri[active]
        tri_nodes = self.B.faces[tri]
        z = torch.cat([points[active], self.B.x[tri_nodes].reshape(-1, 9)], dim=1)
        kA = self.stiffness * self.A.nodal_area[nodes]
        g = self.offset[active]
        energy = 0.5 * kA * (sd[active] - g) ** 2
        arange = torch.arange(3, device=z.device)
        dofs = torch.cat([self.solver.dof_offset[id(self.A)] + 3 * nodes[:, None] + arange,
                          (self.solver.dof_offset[id(self.B)] + 3 * tri_nodes[:, :, None] + arange).reshape(-1, 9)],
                         dim=1)
        return [(dofs, energy, _ce_g(z, g, kA), _ce_h(z, g, kA) if tangent else None)]


def _tie_energy(z, w, r0, kA):
    """z = [node x (3), triangle vertices (9)]."""
    x, tri = z[:3], z[3:].reshape(3, 3)
    d = x - w @ tri - r0
    return 0.5 * kA * (d * d).sum()


_tie_e, _tie_g, _tie_h = vmap(_tie_energy), vmap(grad(_tie_energy)), vmap(hessian(_tie_energy))


class SurfaceTie:
    def __init__(self, body, nodes, solid, stiffness: float, max_distance: float):
        """Bond `nodes` of `body` (a membrane) to the surface of `solid` (nodes within max_distance)."""
        self.body, self.solid, self.stiffness = body, solid, float(stiffness)
        nodes = torch.as_tensor(np.asarray(nodes, np.int64), device=body.device)
        dist, q, tri = closest_on_mesh(body.X[nodes], solid.X, solid.faces, max_distance)
        keep = torch.isfinite(dist)
        self.nodes, q, tri = nodes[keep], q[keep], tri[keep]
        self.tri_nodes = solid.faces[tri]
        a, b, c = (solid.X[self.tri_nodes[:, k]] for k in range(3))
        self.w = _barycentric(q, a, b, c)
        self.r0 = body.X[self.nodes] - q
        self.kA = self.stiffness * body.nodal_area[self.nodes]

    def bind(self, solver):
        self.solver = solver

    def set_stiffness(self, k):
        self.stiffness = float(k)
        self.kA = self.stiffness * self.body.nodal_area[self.nodes]

    def terms(self, tangent=True):
        if not len(self.nodes):
            return []
        z = torch.cat([self.body.x[self.nodes], self.solid.x[self.tri_nodes].reshape(-1, 9)], dim=1)
        arange = torch.arange(3, device=z.device)
        dofs = torch.cat([self.solver.dof_offset[id(self.body)] + 3 * self.nodes[:, None] + arange,
                          (self.solver.dof_offset[id(self.solid)] + 3 * self.tri_nodes[:, :, None] + arange
                           ).reshape(-1, 9)], dim=1)
        args = (z, self.w, self.r0, self.kA)
        return [(dofs, _tie_e(*args), _tie_g(*args), _tie_h(*args) if tangent else None)]


def _barycentric(q, a, b, c):
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return torch.stack([1.0 - v - w, v, w], dim=1)
