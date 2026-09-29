"""
Contact between membranes/shells (between different sheets; a sheet folding onto itself is not
included).

Every sheet is a mid-surface with a thickness. A node of sheet A touches sheet B when it comes
closer than h = t_A/2 + t_B/2 to B's mid-surface (h is reduced where the sheets already start
closer, e.g. where they meet at a frame). Penalty energy for every node-triangle pair within
contact range, in the penetration p = h - d (d = distance from the node to the closest point of
that triangle):

    E = k * A_node * psi(p),   psi = p^3 / (6 delta)                      for 0 < p < delta
                                   = p^2 / 2 - delta p / 2 + delta^2 / 6    for p >= delta

i.e. the ordinary quadratic penalty, with the stiffness ramped up over the first delta = SMOOTHING * h
of penetration. Every pair's energy switches on smoothly (C2), so nodes coming into contact do not
make Newton's tangent jump. All triangles within range take part, not only the closest one: with
only the closest, the contact force jumps from one triangle's vertices to its neighbour's when a
node slides across an edge, and Newton cycles. (Near an edge a node therefore feels two triangles;
with a stiff penalty this changes the separation only negligibly.)

Gradients and Hessians are exact over the node and the triangle's three vertices (torch.func), so
the Newton tangent stays consistent. Both directions (A's nodes on B, B's nodes on A) are used.

A penalty alone can be stepped over: one large step can carry a node straight through the other
sheet to where the penalty is zero again. So every step is bounded conservatively (as in IPC's
continuous collision handling): a node at distance d from a triangle can only reach it if the two
approach by d, and they approach by at most |node motion| + max |triangle vertex motion|. Steps
are shortened so that no nearby node-triangle pair closes more than 90 % of its current distance,
so a node can never pass through another sheet.

Pair searches use candidate lists (Verlet lists): each query radius R keeps the node-triangle pairs
within R + 2 skin, rebuilt once a node of either sheet has moved more than skin, which keeps every
pair within R (node and triangle close by at most 2 skin in between). Exact, and much cheaper than
testing every node against every triangle on each evaluation.
"""
import math

import torch
from torch.func import grad, hessian, vmap

from .contact import _closest_point_pairs

DTYPE = torch.float64


SMOOTHING = 0.25


def _pair_energy(x, kA, h, delta):
    """x = [node (3), triangle vertices a, b, c (9)]."""
    p, a, b, c = x[0:3], x[3:6], x[6:9], x[9:12]
    q, _ = _closest_point_pairs(p[None], a[None], b[None], c[None])
    pen = h - torch.linalg.norm(p - q[0])
    ramp = pen.clamp_min(0.0) ** 3 / (6.0 * delta)
    full = 0.5 * pen ** 2 - 0.5 * delta * pen + delta ** 2 / 6.0
    return kA * torch.where(pen < delta, ramp, full)


_energy = vmap(_pair_energy)
_gradient = vmap(grad(_pair_energy))
_hessian = vmap(hessian(_pair_energy))


def closest_on_mesh(points, vertices, faces, max_distance, chunk_elements=2_000_000):
    """
    Closest point on a triangle mesh for every point: (distance, closest point, triangle index).
    Points further than max_distance from every triangle get distance inf.
    """
    n, device = len(points), points.device
    dist = torch.full((n,), math.inf, dtype=DTYPE, device=device)
    closest = torch.zeros((n, 3), dtype=DTYPE, device=device)
    triangle = torch.zeros(n, dtype=torch.long, device=device)
    if n == 0 or len(faces) == 0:
        return dist, closest, triangle
    tri = vertices[faces]
    lo, hi = tri.min(dim=1).values - max_distance, tri.max(dim=1).values + max_distance
    chunk = max(1, chunk_elements // len(faces))
    for start in range(0, n, chunk):
        p = points[start:start + chunk]
        near = ((p[:, None] >= lo) & (p[:, None] <= hi)).all(-1)
        pi, ti = torch.nonzero(near, as_tuple=True)
        if not len(pi):
            continue
        q, _ = _closest_point_pairs(p[pi], tri[ti, 0], tri[ti, 1], tri[ti, 2])
        d2 = ((p[pi] - q) ** 2).sum(-1)
        order = torch.argsort(d2, stable=True)
        order = order[torch.argsort(pi[order], stable=True)]
        first = torch.ones_like(order, dtype=torch.bool)
        first[1:] = pi[order][1:] != pi[order][:-1]
        best = order[first]
        best = best[d2[best] <= max_distance ** 2]
        idx = start + pi[best]
        dist[idx], closest[idx], triangle[idx] = d2[best].sqrt(), q[best], ti[best]
    return dist, closest, triangle


def pairs_within(points, vertices, faces, max_distance, chunk_elements=2_000_000):
    """All (point index, triangle index, distance, closest point) pairs closer than max_distance."""
    device = points.device
    empty = (torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, dtype=torch.long, device=device),
             torch.zeros(0, dtype=DTYPE, device=device), torch.zeros((0, 3), dtype=DTYPE, device=device))
    if len(points) == 0 or len(faces) == 0:
        return empty
    tri = vertices[faces]
    lo, hi = tri.min(dim=1).values - max_distance, tri.max(dim=1).values + max_distance
    chunk = max(1, chunk_elements // len(faces))
    parts = []
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        near = ((p[:, None] >= lo) & (p[:, None] <= hi)).all(-1)
        pi, ti = torch.nonzero(near, as_tuple=True)
        if not len(pi):
            continue
        q, _ = _closest_point_pairs(p[pi], tri[ti, 0], tri[ti, 1], tri[ti, 2])
        d = (p[pi] - q).norm(dim=1)
        keep = d < max_distance
        parts.append((start + pi[keep], ti[keep], d[keep], q[keep]))
    if not parts:
        return empty
    return tuple(torch.cat(x) for x in zip(*parts))


def _face_normals(x, faces):
    tri = x[faces]
    n = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-300)


class ShellContact:
    def __init__(self, shells, stiffness: float, search_distance: float, pairs=None):
        """
        shells          : the Shell objects that can touch each other.
        stiffness       : penalty stiffness k (pressure per unit penetration).
        search_distance : pairs within this distance are checked when bounding a step; at least the
                          largest relative motion of a node and a triangle in one step.
        pairs           : instead of all shell pairs, explicit (A, A's node indices, B, B's face indices, h)
                          entries, e.g. the two opposite walls inside a tube (A may be B: the node and
                          face sets then keep apart parts of one body; pairs that start closer than h,
                          such as a node on a triangle it belongs to, never become active).
        """
        self.stiffness = stiffness
        self.search_distance = search_distance
        self.pairs = []
        if pairs is not None:
            for A, nodes, B, faces, h in pairs:
                nodes = torch.as_tensor(nodes, device=A.device)
                nodes = nodes[~A.fixed[nodes]]
                faces = B.faces[torch.as_tensor(faces, device=B.device)]
                if not len(nodes) or not len(faces) or h <= 0:
                    continue
                rest, _, _ = closest_on_mesh(A.X[nodes], B.X, faces, h)
                self.pairs.append({"A": A, "B": B, "nodes": nodes, "faces": faces, "h_max": h,
                                   "h": torch.clamp(rest, max=h), "skin": h, "lists": {}})
            return
        for A in shells:
            for B in shells:
                if A is B:
                    continue
                h = _half_thickness(A) + _half_thickness(B)
                if h <= 0:
                    continue
                nodes = torch.nonzero(~A.fixed).flatten()
                if not len(nodes):
                    continue
                rest, _, _ = closest_on_mesh(A.X[nodes], B.X, B.faces, h)
                self.pairs.append({"A": A, "B": B, "nodes": nodes, "faces": B.faces, "h_max": h,
                                   "h": torch.clamp(rest, max=h), "skin": h, "lists": {}})

    def __bool__(self):
        return bool(self.pairs)

    def _pairs_within(self, pair, radius):
        """pairs_within(A's nodes, B, radius) at the current state, from the pair's candidate list."""
        A, B, nodes, skin, faces = pair["A"], pair["B"], pair["nodes"], pair["skin"], pair["faces"]
        points = A.x[nodes]
        cached = pair["lists"].get(radius)
        if cached is None or (points - cached["a"]).norm(dim=1).max().item() > skin or                 (B.x - cached["b"]).norm(dim=1).max().item() > skin:
            pi, t, _, _ = pairs_within(points, B.x, faces, radius + 2.0 * skin)
            cached = pair["lists"][radius] = {"a": points.clone(), "b": B.x.clone(), "pi": pi, "t": t}
        pi, t = cached["pi"], cached["t"]
        if len(pi):  # cheap bounding-box test first, the exact distance only for what is left
            tri = B.x[faces[t]]
            p = points[pi]
            near = ((p >= tri.min(dim=1).values - radius) & (p <= tri.max(dim=1).values + radius)).all(dim=1)
            pi, t, tri = pi[near], t[near], tri[near]
        if not len(pi):
            return pi, t, torch.zeros(0, dtype=DTYPE, device=points.device), points[:0]
        q, _ = _closest_point_pairs(points[pi], tri[:, 0], tri[:, 1], tri[:, 2])
        d = (points[pi] - q).norm(dim=1)
        keep = d < radius
        return pi[keep], t[keep], d[keep], q[keep]

    def safe_step(self, displacement: dict) -> float:
        """
        Largest fraction (<= 1) of a step that keeps every nearby node-triangle pair at least 10 % of
        its current distance apart. displacement: {id(shell): (n_nodes, 3) node motion of the full step}.
        """
        fraction = 1.0
        for pair in self.pairs:
            A, B, nodes, faces = pair["A"], pair["B"], pair["nodes"], pair["faces"]
            radius = pair["h_max"] + self.search_distance
            # The distance to a triangle's bounding box is a lower bound d_lb <= d, so a pair with
            # 0.9 d_lb >= fraction * closing cannot shorten the step: only the others need the exact
            # distance (same result as testing every pair within radius, at a fraction of the cost).
            points, tri = A.x[nodes], B.x[faces]
            gap = ((tri.min(dim=1).values - points[:, None]).clamp_min(0) ** 2 +
                   (points[:, None] - tri.max(dim=1).values).clamp_min(0) ** 2).sum(-1).sqrt()
            node_motion = displacement[id(A)][nodes].norm(dim=1)
            tri_motion = displacement[id(B)][faces].norm(dim=2).max(dim=1).values
            closing = node_motion[:, None] + tri_motion[None, :]
            pi, t = torch.nonzero((gap < radius) & (0.9 * gap < fraction * closing), as_tuple=True)
            if A is B and len(pi):  # within one body a node never limits the step on its own triangles
                own = (faces[t] == nodes[pi][:, None]).any(dim=1)
                pi, t = pi[~own], t[~own]
            if not len(pi):
                continue
            q, _ = _closest_point_pairs(points[pi], tri[t, 0], tri[t, 1], tri[t, 2])
            d = (points[pi] - q).norm(dim=1)
            within = d < radius
            d, closing = d[within], closing[pi[within], t[within]]
            if not len(d):
                continue
            limit = torch.where(closing > 0, 0.9 * d / closing.clamp_min(1e-300), torch.full_like(d, math.inf))
            fraction = min(fraction, limit.min().item())
        return fraction

    def terms(self, tangent: bool = True):
        """
        [(A, B, node indices of A, triangle node indices of B, energy, gradient (m, 12), hessian)]
        for the node-triangle pairs in contact at the current state.
        """
        out = []
        for pair in self.pairs:
            A, B, nodes = pair["A"], pair["B"], pair["nodes"]
            points = A.x[nodes]
            pi, t, d, q = self._pairs_within(pair, pair["h_max"])
            active = d < pair["h"][pi]
            if not active.any():
                continue
            pi, t = pi[active], t[active]
            tri_nodes = pair["faces"][t]
            x = torch.cat([points[pi], B.x[tri_nodes].reshape(-1, 9)], dim=1)
            kA = self.stiffness * A.nodal_area[nodes[pi]]
            h = pair["h"][pi]
            delta = SMOOTHING * h.clamp_min(1e-12)
            out.append((A, B, nodes[pi], tri_nodes, _energy(x, kA, h, delta), _gradient(x, kA, h, delta),
                        _hessian(x, kA, h, delta) if tangent else None))
        return out


def _half_thickness(shell):
    offset = getattr(shell, "contact_offset", None)
    return float(offset) if offset else 0.5 * float(shell.thickness)
