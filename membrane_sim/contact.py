"""
Rigid obstacles as closed triangle meshes, queried exactly (no SDF grid).

For each shell node the signed distance to the obstacle surface is computed from the
closest point on the triangles (bounding-box broad phase); the sign comes from the
angle-weighted pseudo-normal of the closest feature, which is exact for closed meshes.
Contact is a penalty energy per node,
    E_c = 1/2 * k * A_node * min(sd - offset, 0)^2,
so k has units of pressure per penetration depth and does not depend on the mesh.
"""
import math

import numpy as np
import torch

DTYPE = torch.float64

# Rest overlap deeper than this x sqrt(nodal area) (~ a quarter of the local element size) is a
# real interference (e.g. a rigid body modelled pushed into a soft part), not CAD faceting.
INTERFERENCE_TOLERANCE = 0.25


def interference_offsets(rest_sd, neutral, final, nodal_area):
    """
    Contact offsets (start, final) per node for an interference ramp. Nodes that overlap the
    obstacle at rest by more than the faceting tolerance start neutral (offset = rest distance, no
    force) and end at `final`, so the solver pushes the soft part out of the obstacle gradually;
    every other node keeps `neutral` throughout. Returns (start, final, interfering mask).
    """
    real = rest_sd < -INTERFERENCE_TOLERANCE * nodal_area.sqrt()
    return torch.where(real, rest_sd, neutral), torch.where(real, final, neutral), real


def insertion_path(obstacle, points, clearance, interfering, scan=64, tolerance=1e-6):
    """
    How to insert a rigid obstacle that is modelled pushed deep into a soft part: the shortest rigid
    translation, along a coordinate axis or the depth-weighted mean push-out direction, that takes it clear
    of every point (signed distance >= that point's clearance). Pushing each interfering node out through
    its nearest face fails for deep overlaps (nodes on the far side of the part are sent through the wrong
    face); moving the obstacle back along this path and ramping it into place pushes the part ahead of it,
    as in assembly. Returns (direction the obstacle moves in when inserted, distance) or None.
    """
    sd, gradient = obstacle.signed_distance(points[interfering], max_distance=math.inf)
    depth = (clearance[interfering] - sd).clamp_min(0)
    candidates = [torch.eye(3, dtype=DTYPE, device=points.device)[k] * s for k in range(3) for s in (1.0, -1.0)]
    mean = (depth[:, None] * gradient).sum(0)
    if mean.norm() > 1e-12:
        candidates.append(mean / mean.norm())
    preferred = []
    if obstacle.insert_direction is not None:   # a chosen axis (e.g. the set-up move) wins whenever it clears;
        n = torch.as_tensor(obstacle.insert_direction, dtype=DTYPE, device=points.device)
        preferred = [n, -n]                     # either sense: a body moved away from a part still pushes it out
    size = (obstacle.upper - obstacle.lower).norm().item() + (depth.max().item() if len(depth) else 0.0)
    tol = tolerance * max(size, 1.0)

    def clear(direction, d):  # obstacle moved back by d against `direction`
        s, _ = obstacle.signed_distance(points + d * direction, max_distance=math.inf)
        return bool((s >= clearance - tol).all())

    def shortest(directions):
        best = None
        for direction in directions:
            steps = torch.linspace(0.0, 2.0 * size, scan + 1).tolist()[1:]
            hi = next((d for d in steps if clear(direction, d)), None)
            if hi is None or (best is not None and hi - steps[0] >= best[1]):
                continue
            lo = max(hi - steps[0], 0.0)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if clear(direction, mid) else (mid, hi)
            if best is None or hi < best[1]:
                best = (direction, hi)
        return best

    return (shortest(preferred) if preferred else None) or shortest(candidates)


class Obstacle:
    def __init__(self, vertices, faces, inverted: bool = False, color: str = "grey", device="cpu"):
        """
        vertices, faces : closed, consistently oriented mesh with outward normals.
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
        self._lo, self._hi = tri.min(dim=1).values, tri.max(dim=1).values
        self.lower = self.vertices.min(dim=0).values
        self.upper = self.vertices.max(dim=0).values
        # Rigid translation of the whole obstacle; queries are made at (points - shift). The solver sets it
        # (NewtonSolver.interference) to insert an obstacle modelled pushed into a soft part, along
        # `insert_direction` when given (e.g. the direction it was moved in), else its shortest clearing path.
        self.shift = torch.zeros(3, dtype=DTYPE, device=self.device)
        self.insert_direction = None
        self._build_pseudonormals()

    def _build_pseudonormals(self):
        """Angle-weighted pseudo-normals (Baerentzen & Aanaes): the sign of (p - q) . N at the
        closest feature q is exact for closed meshes."""
        F = self.faces.cpu().numpy()
        V = self.vertices.cpu().numpy()
        tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)

        vertex_n = np.zeros_like(V)
        for k in range(3):
            e1 = tri[:, (k + 1) % 3] - tri[:, k]
            e2 = tri[:, (k + 2) % 3] - tri[:, k]
            cos = np.einsum("ij,ij->i", e1, e2) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1))
            np.add.at(vertex_n, F[:, k], np.arccos(np.clip(cos, -1, 1))[:, None] * n)
        vertex_n /= np.linalg.norm(vertex_n, axis=1, keepdims=True)

        edge_sum = {}
        for f, (a, b, c) in enumerate(F):
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(i, j), max(i, j))
                edge_sum[key] = edge_sum.get(key, 0.0) + n[f]
        edge_n = np.stack([[edge_sum[(min(i, j), max(i, j))] for i, j in ((a, b), (b, c), (c, a))]
                           for a, b, c in F])
        edge_n /= np.linalg.norm(edge_n, axis=2, keepdims=True)

        # feature normals per face: [face, edge ab, edge bc, edge ca, vertex a, vertex b, vertex c]
        features = np.concatenate([n[:, None], edge_n, vertex_n[F]], axis=1)
        self._feature_normals = torch.as_tensor(features, dtype=DTYPE, device=self.device)

    def signed_distance(self, points: torch.Tensor, max_distance: float, chunk_elements: int = 2_000_000,
                        hessian: bool = False):
        """
        Signed distance (negative inside the solid) and its gradient for points (N, 3), plus its
        Hessian (N, 3, 3) if requested (non-zero where the closest feature is an edge or vertex).
        Points that are certainly further than max_distance outside the solid get +inf.
        """
        n = points.shape[0]
        sd = torch.full((n,), math.inf, dtype=DTYPE, device=points.device)
        gradient = torch.zeros((n, 3), dtype=DTYPE, device=points.device)
        H = torch.zeros((n, 3, 3), dtype=DTYPE, device=points.device) if hessian else None
        points = points - self.shift

        if self.inverted:
            candidates = torch.arange(n, device=points.device)
        else:
            near = ((points >= self.lower - max_distance) & (points <= self.upper + max_distance)).all(dim=1)
            candidates = torch.nonzero(near).flatten()
        if candidates.numel() == 0:
            return (sd, gradient, H) if hessian else (sd, gradient)

        n_faces = self.faces.shape[0]
        chunk = max(1, chunk_elements // max(n_faces, self.vertices.shape[0]))
        for start in range(0, candidates.numel(), chunk):
            idx = candidates[start:start + chunk]
            p = points[idx]

            pi, ti = self._candidates(p, margin=0.0)
            out = self._closest(p, pi, ti, hessian)
            sd[idx], gradient[idx] = out[0], out[1]
            if hessian:
                H[idx] = out[2]

        if self.inverted:
            sd, gradient = -sd, -gradient
            H = -H if hessian else None
        return (sd, gradient, H) if hessian else (sd, gradient)

    def _candidates(self, p, margin: float):
        """Broad phase: (point, triangle) pairs that can hold a point's closest triangle, also after
        every point and nothing else moved by up to margin/2. The nearest mesh vertex bounds the
        distance d; triangles whose bounding box is further than d + margin are left out."""
        bound = torch.cdist(p, self.vertices).min(dim=1).values + margin
        box2 = ((self._lo - p[:, None]).clamp_min(0) ** 2 + (p[:, None] - self._hi).clamp_min(0) ** 2).sum(-1)
        return torch.nonzero(box2 <= (bound ** 2)[:, None] * (1 + 1e-9) + 1e-18, as_tuple=True)

    def _closest(self, p, pi, ti, hessian: bool):
        """Narrow phase: signed distance (solid = inside the mesh), gradient and Hessian of every
        point p[i] from its closest triangle among the pairs (pi, ti). Every point needs a pair."""
        q, feature = _closest_point_pairs(p[pi], self._a[ti], self._b[ti], self._c[ti])
        d2 = ((p[pi] - q) ** 2).sum(-1)
        order = torch.argsort(d2, stable=True)
        order = order[torch.argsort(pi[order], stable=True)]
        first = torch.ones_like(order, dtype=torch.bool)
        first[1:] = pi[order][1:] != pi[order][:-1]
        best = order[first]                                   # one pair per point, in point order

        q, dist = q[best], d2[best].sqrt()
        normal = self._feature_normals[ti[best], feature[best]]
        diff = p - q
        sign = torch.where((diff * normal).sum(-1) < 0, -1.0, 1.0).to(DTYPE)
        unit = torch.where((dist > 1e-12)[:, None], diff / dist.clamp_min(1e-300)[:, None], sign[:, None] * normal)
        if not hessian:
            return sign * dist, sign[:, None] * unit, None

        # d2|p - q|/dp2: 0 on a face, (I - uu^T - tt^T)/d on an edge, (I - uu^T)/d at a vertex
        f = feature[best]
        eye = torch.eye(3, dtype=DTYPE, device=p.device).expand(len(p), 3, 3)
        proj = eye - unit[:, :, None] * unit[:, None, :]
        tri = torch.stack([self._a[ti[best]], self._b[ti[best]], self._c[ti[best]]], dim=1)
        start_v = torch.gather(tri, 1, ((f - 1).clamp(0, 2))[:, None, None].expand(-1, 1, 3))[:, 0]
        end_v = torch.gather(tri, 1, (f % 3)[:, None, None].expand(-1, 1, 3))[:, 0]
        t = end_v - start_v
        t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-300)
        on_edge = ((f >= 1) & (f <= 3))[:, None, None]
        proj = torch.where(on_edge, proj - t[:, :, None] * t[:, None, :], proj)
        curved = ((f >= 1) & (dist > 1e-12))[:, None, None]
        H = torch.where(curved, sign[:, None, None] * proj / dist.clamp_min(1e-300)[:, None, None],
                        torch.zeros_like(proj))
        return sign * dist, sign[:, None] * unit, H

    def to(self, device):
        self.__init__(self.vertices.cpu().numpy(), self.faces.cpu().numpy(), self.inverted, self.color, device)
        return self


class ObstacleField:
    """Union of obstacles: the signed distance is the minimum over all obstacles."""

    def __init__(self, obstacles):
        self.obstacles = list(obstacles)

    def signed_distance(self, points: torch.Tensor, max_distance: float, hessian: bool = False):
        sd = torch.full((points.shape[0],), math.inf, dtype=DTYPE, device=points.device)
        gradient = torch.zeros_like(points)
        H = torch.zeros((points.shape[0], 3, 3), dtype=DTYPE, device=points.device) if hessian else None
        for obstacle in self.obstacles:
            out = obstacle.signed_distance(points, max_distance, hessian=hessian)
            closer = out[0] < sd
            sd = torch.where(closer, out[0], sd)
            gradient = torch.where(closer[:, None], out[1], gradient)
            if hessian:
                H = torch.where(closer[:, None, None], out[2], H)
        return (sd, gradient, H) if hessian else (sd, gradient)


class CachedSignedDistance:
    """
    ObstacleField.signed_distance for one fixed set of moving points (a shell's contact nodes),
    with candidate lists (Verlet lists) instead of a search over every triangle on every call.

    At a rebuild each point keeps the triangles within its current distance + 2 skin; the true
    closest triangle is then among them for as long as no point has moved more than skin, so the
    result is exactly that of the full search. Points further than max_distance + skin from every
    obstacle are left out (+inf) until the next rebuild.
    """

    def __init__(self, field: ObstacleField, max_distance: float, skin: float):
        self.field, self.max_distance, self.skin = field, float(max_distance), float(skin)
        self._reference = None
        self.rebuilds = 0

    def _rebuild(self, points):
        self._reference = points.clone()
        self._shifts = [o.shift.clone() for o in self.field.obstacles]
        self.rebuilds += 1
        self._lists = []
        for obstacle in self.field.obstacles:
            local = points - obstacle.shift
            if obstacle.inverted:
                idx = torch.arange(len(points), device=points.device)
            else:
                reach = self.max_distance + self.skin
                near = ((local >= obstacle.lower - reach) & (local <= obstacle.upper + reach)).all(dim=1)
                idx = torch.nonzero(near).flatten()
            if not len(idx):
                continue
            pi, ti = obstacle._candidates(local[idx], margin=2.0 * self.skin)
            sd, _, _ = obstacle._closest(local[idx], pi, ti, hessian=False)
            if obstacle.inverted:
                sd = -sd
            keep = sd <= self.max_distance + self.skin
            if not keep.any():
                continue
            renumber = torch.full((len(idx),), -1, dtype=torch.long, device=points.device)
            renumber[keep] = torch.arange(int(keep.sum()), device=points.device)
            pair = keep[pi]
            self._lists.append((obstacle, idx[keep], renumber[pi[pair]], ti[pair]))

    def signed_distance(self, points: torch.Tensor, hessian: bool = False):
        if self._reference is None or len(points) != len(self._reference) or \
                (points - self._reference).norm(dim=1).max().item() > self.skin or \
                any(not torch.equal(o.shift, s) for o, s in zip(self.field.obstacles, self._shifts)):
            self._rebuild(points)
        n = len(points)
        sd = torch.full((n,), math.inf, dtype=DTYPE, device=points.device)
        gradient = torch.zeros_like(points)
        H = torch.zeros((n, 3, 3), dtype=DTYPE, device=points.device) if hessian else None
        for obstacle, idx, pi, ti in self._lists:
            o_sd, o_gradient, o_H = obstacle._closest(points[idx] - obstacle.shift, pi, ti, hessian)
            if obstacle.inverted:
                o_sd, o_gradient = -o_sd, -o_gradient
                o_H = -o_H if hessian else None
            closer = o_sd < sd[idx]
            sd[idx] = torch.where(closer, o_sd, sd[idx])
            gradient[idx] = torch.where(closer[:, None], o_gradient, gradient[idx])
            if hessian:
                H[idx] = torch.where(closer[:, None, None], o_H, H[idx])
        return (sd, gradient, H) if hessian else (sd, gradient)


# -----------------------------
# Geometry kernels
# -----------------------------

def _closest_point_pairs(p, a, b, c):
    """
    Closest point on triangle (a, b, c) to p, for matching rows (Ericson, Real-Time Collision
    Detection 5.1.5). Returns q (n, 3) and the feature it lies on:
    0 face, 1 edge ab, 2 edge bc, 3 edge ca, 4 vertex a, 5 vertex b, 6 vertex c.
    """
    def dot(u, v):
        return (u * v).sum(-1)

    ab, ac, ap = b - a, c - a, p - a
    bp, cp = p - b, p - c
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def safe(num, den):
        return num / torch.where(den.abs() > 1e-300, den, torch.ones_like(den))

    denom = va + vb + vc
    q = a + safe(vb, denom)[:, None] * ab + safe(vc, denom)[:, None] * ac
    feature = torch.zeros(len(p), dtype=torch.long, device=p.device)

    # Later assignments have priority, so apply the regions in reverse order of Ericson's tests
    regions = [
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
         b + safe(d4 - d3, (d4 - d3) + (d5 - d6))[:, None] * (c - b), 2),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + safe(d2, d2 - d6)[:, None] * ac, 3),
        ((d6 >= 0) & (d5 <= d6), c, 6),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + safe(d1, d1 - d3)[:, None] * ab, 1),
        ((d3 >= 0) & (d4 <= d3), b, 5),
        ((d1 <= 0) & (d2 <= 0), a, 4),
    ]
    for mask, point, code in regions:
        q = torch.where(mask[:, None], point, q)
        feature = torch.where(mask, code, feature)
    return q, feature


def _winding_number(p, a, b, c):
    """Generalised winding number (Van Oosterom-Strackee solid angles): ~1 inside, ~0 outside."""
    A, B, C = a - p[:, None, :], b - p[:, None, :], c - p[:, None, :]
    la, lb, lc = A.norm(dim=-1), B.norm(dim=-1), C.norm(dim=-1)
    num = (A * torch.linalg.cross(B, C)).sum(-1)
    den = la * lb * lc + (A * B).sum(-1) * lc + (A * C).sum(-1) * lb + (B * C).sum(-1) * la
    return (2.0 * torch.atan2(num, den)).sum(-1) / (4.0 * math.pi)
