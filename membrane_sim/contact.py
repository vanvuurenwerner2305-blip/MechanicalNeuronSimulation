"""
Rigid obstacles as closed triangle meshes, queried exactly (no SDF grid).

For each shell node the signed distance to the obstacle surface is computed from the
closest point on the triangles; the sign comes from the generalised winding number.
Contact is a penalty energy per node,
    E_c = 1/2 * k * A_node * min(sd - offset, 0)^2,
so k has units of pressure per penetration depth and does not depend on the mesh.
"""
import math

import numpy as np
import torch

DTYPE = torch.float64


class Obstacle:
    def __init__(self, vertices, faces, inverted: bool = False, color: str = "grey", device="cpu"):
        """
        vertices, faces : closed mesh with outward normals.
        inverted        : if True the solid is *outside* the mesh (e.g. a container box whose
                          walls the membranes must stay inside).
        """
        self.device = torch.device(device)
        self.vertices = torch.as_tensor(np.asarray(vertices, float), dtype=DTYPE, device=self.device)
        self.faces = torch.as_tensor(np.asarray(faces, np.int64), device=self.device)
        self.inverted = inverted
        self.color = color

        tri = self.vertices[self.faces]
        self._a, self._b, self._c = tri[:, 0], tri[:, 1], tri[:, 2]
        n = torch.linalg.cross(self._b - self._a, self._c - self._a)
        self._normals = n / n.norm(dim=1, keepdim=True)
        self.lower = self.vertices.min(dim=0).values
        self.upper = self.vertices.max(dim=0).values

    def signed_distance(self, points: torch.Tensor, max_distance: float, chunk_elements: int = 2_000_000):
        """
        Signed distance (negative inside the solid) and its gradient for points (N, 3).
        Points that are certainly further than max_distance outside the solid get +inf.
        """
        n = points.shape[0]
        sd = torch.full((n,), math.inf, dtype=DTYPE, device=points.device)
        gradient = torch.zeros((n, 3), dtype=DTYPE, device=points.device)

        if self.inverted:
            candidates = torch.arange(n, device=points.device)
        else:
            near = ((points >= self.lower - max_distance) & (points <= self.upper + max_distance)).all(dim=1)
            candidates = torch.nonzero(near).flatten()
        if candidates.numel() == 0:
            return sd, gradient

        chunk = max(1, chunk_elements // self.faces.shape[0])
        for start in range(0, candidates.numel(), chunk):
            idx = candidates[start:start + chunk]
            p = points[idx]
            q = _closest_points_on_triangles(p, self._a, self._b, self._c)  # (n, T, 3)
            d2, j = ((p[:, None, :] - q) ** 2).sum(-1).min(dim=1)
            q = q[torch.arange(len(idx), device=p.device), j]
            dist = d2.sqrt()

            inside = _winding_number(p, self._a, self._b, self._c) > 0.5
            sign = 1.0 - 2.0 * inside.to(DTYPE)
            direction = torch.where((dist > 1e-12)[:, None],
                                    sign[:, None] * (p - q) / dist.clamp_min(1e-300)[:, None],
                                    self._normals[j])
            sd[idx] = sign * dist
            gradient[idx] = direction

        if self.inverted:
            sd, gradient = -sd, -gradient
        return sd, gradient

    def to(self, device):
        self.__init__(self.vertices.cpu().numpy(), self.faces.cpu().numpy(), self.inverted, self.color, device)
        return self


class ObstacleField:
    """Union of obstacles: the signed distance is the minimum over all obstacles."""

    def __init__(self, obstacles):
        self.obstacles = list(obstacles)

    def signed_distance(self, points: torch.Tensor, max_distance: float):
        sd = torch.full((points.shape[0],), math.inf, dtype=DTYPE, device=points.device)
        gradient = torch.zeros_like(points)
        for obstacle in self.obstacles:
            d, g = obstacle.signed_distance(points, max_distance)
            closer = d < sd
            sd = torch.where(closer, d, sd)
            gradient = torch.where(closer[:, None], g, gradient)
        return sd, gradient


# -----------------------------
# Geometry kernels
# -----------------------------

def _closest_points_on_triangles(p, a, b, c):
    """Closest point on every triangle (T) for every point (n): returns (n, T, 3)."""
    p = p[:, None, :]
    ab, ac = b - a, c - a
    n = torch.linalg.cross(ab, ac)
    nn = (n * n).sum(-1)
    ap = p - a
    w_c = (torch.linalg.cross(ab.expand_as(ap), ap) * n).sum(-1) / nn
    w_b = (torch.linalg.cross(ap, ac.expand_as(ap)) * n).sum(-1) / nn
    inside = (w_b >= 0) & (w_c >= 0) & (w_b + w_c <= 1)
    q_face = a + w_b[..., None] * ab + w_c[..., None] * ac

    def on_segment(s0, d):
        s = (((p - s0) * d).sum(-1) / (d * d).sum(-1)).clamp(0.0, 1.0)
        q = s0 + s[..., None] * d
        return q, ((p - q) ** 2).sum(-1)

    q1, d1 = on_segment(a, ab)
    q2, d2 = on_segment(b, c - b)
    q3, d3 = on_segment(c, a - c)
    q_edge = torch.where(((d1 <= d2) & (d1 <= d3))[..., None], q1, torch.where((d2 <= d3)[..., None], q2, q3))
    return torch.where(inside[..., None], q_face, q_edge)


def _winding_number(p, a, b, c):
    """Generalised winding number (Van Oosterom-Strackee solid angles): ~1 inside, ~0 outside."""
    A, B, C = a - p[:, None, :], b - p[:, None, :], c - p[:, None, :]
    la, lb, lc = A.norm(dim=-1), B.norm(dim=-1), C.norm(dim=-1)
    num = (A * torch.linalg.cross(B, C)).sum(-1)
    den = la * lb * lc + (A * B).sum(-1) * lc + (A * C).sum(-1) * lb + (B * C).sum(-1) * la
    return (2.0 * torch.atan2(num, den)).sum(-1) / (4.0 * math.pi)
