"""
Mesh helpers.

Every mesh is a pair ``(vertices, faces)``: ``vertices`` is a (V, 3) float array and
``faces`` a (F, 3) int array. Faces are ordered counter-clockwise when seen from the
side the surface normal points to, so ``normal = (x1 - x0) x (x2 - x0)``.
Closed meshes (obstacles) must have outward-pointing normals.
"""
import numpy as np
from scipy.spatial import Delaunay


# -----------------------------
# Membrane (open surface) meshes
# -----------------------------

def rectangle_mesh(corner, edge_u, edge_v, nu: int, nv: int):
    """
    Structured triangulation of the parallelogram corner + s*edge_u + t*edge_v, s,t in [0,1].
    Diagonals alternate between cells to avoid a directional bias. Normal = edge_u x edge_v.
    """
    corner = np.asarray(corner, dtype=float)
    edge_u = np.asarray(edge_u, dtype=float)
    edge_v = np.asarray(edge_v, dtype=float)

    s = np.linspace(0.0, 1.0, nu + 1)
    t = np.linspace(0.0, 1.0, nv + 1)
    S, T = np.meshgrid(s, t, indexing="ij")
    vertices = corner + S[..., None] * edge_u + T[..., None] * edge_v
    vertices = vertices.reshape(-1, 3)

    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = (i + 1) * (nv + 1) + j
            c = (i + 1) * (nv + 1) + j + 1
            d = i * (nv + 1) + j + 1
            if (i + j) % 2 == 0:
                faces += [(a, b, c), (a, c, d)]
            else:
                faces += [(a, b, d), (b, c, d)]
    return vertices, np.asarray(faces, dtype=np.int64)


def disk_mesh(center, normal, radius: float, rings: int):
    """Unstructured triangulation of a disk from concentric rings of 6*k points."""
    center = np.asarray(center, dtype=float)
    e1, e2, n = _plane_basis(normal)

    pts = [np.zeros(2)]
    for k in range(1, rings + 1):
        r = radius * k / rings
        phi = np.linspace(0.0, 2.0 * np.pi, 6 * k, endpoint=False) + 0.5 * np.pi / (6 * k) * (k % 2)
        pts.append(np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1))
    pts2d = np.vstack(pts)

    faces = _orient_ccw(pts2d, Delaunay(pts2d).simplices)
    vertices = center + pts2d[:, :1] * e1 + pts2d[:, 1:] * e2
    return vertices, faces


# -----------------------------
# Obstacle (closed surface) meshes
# -----------------------------

def box_mesh(lower, upper):
    """Closed axis-aligned box with outward normals."""
    (x0, y0, z0), (x1, y1, z1) = np.asarray(lower, float), np.asarray(upper, float)
    vertices = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = np.array([[0, 2, 1], [0, 3, 2],   # z0
                      [4, 5, 6], [4, 6, 7],   # z1
                      [0, 1, 5], [0, 5, 4],   # y0
                      [3, 7, 6], [3, 6, 2],   # y1
                      [0, 4, 7], [0, 7, 3],   # x0
                      [1, 2, 6], [1, 6, 5]],  # x1
                     dtype=np.int64)
    return vertices, faces


def extrude_polygon(polygon, z0: float, z1: float):
    """
    Extrude a simple 2D polygon (M, 2) in the xy-plane from z = z0 to z = z1 into a closed
    prism. This is how 2D obstacle outlines from the old simulator become 3D obstacles.
    """
    poly = _clean_polygon(np.asarray(polygon, dtype=float))
    if _signed_area(poly) < 0:
        poly = poly[::-1]
    m = len(poly)

    vertices = np.vstack([np.column_stack([poly, np.full(m, z0)]),
                          np.column_stack([poly, np.full(m, z1)])])
    cap = triangulate_polygon(poly)
    faces = [cap[:, ::-1], cap + m]  # bottom faces -z, top faces +z
    side = []
    for i in range(m):
        j = (i + 1) % m
        side += [(i, j, j + m), (i, j + m, i + m)]
    faces.append(np.asarray(side, dtype=np.int64))
    return vertices, np.vstack(faces)


def triangulate_polygon(polygon):
    """Ear-clipping triangulation of a simple CCW polygon. Returns (M-2, 3) indices."""
    poly = np.asarray(polygon, dtype=float)
    remaining = list(range(len(poly)))
    triangles = []
    scale = np.ptp(poly, axis=0).max() ** 2

    while len(remaining) > 3:
        for k in range(len(remaining)):
            ia, ib, ic = remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]
            a, b, c = poly[ia], poly[ib], poly[ic]
            turn = _cross2(b - a, c - b)
            if abs(turn) <= 1e-12 * scale:  # collinear vertex: drop it, it bounds no area
                remaining.pop(k)
                break
            if turn < 0:  # reflex vertex
                continue
            if any(_point_in_triangle(poly[j], a, b, c)
                   for j in remaining if j not in (ia, ib, ic)):
                continue
            triangles.append((ia, ib, ic))
            remaining.pop(k)
            break
        else:
            raise ValueError("Ear clipping failed: polygon is not simple.")
    triangles.append(tuple(remaining))
    return np.asarray(triangles, dtype=np.int64)


def closed_mesh_volume(vertices, faces) -> float:
    """Enclosed volume of a closed, outward-oriented mesh."""
    v = np.asarray(vertices, float)[np.asarray(faces)]
    return float(np.einsum("fi,fi->f", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def boundary_nodes(faces, n_vertices: int):
    """Boolean mask of vertices on the open boundary of a mesh."""
    faces = np.asarray(faces)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    mask = np.zeros(n_vertices, dtype=bool)
    mask[unique[counts == 1].ravel()] = True
    return mask


# -----------------------------
# Internal helpers
# -----------------------------

def _plane_basis(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(helper, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2, n


def _orient_ccw(pts2d, tris):
    tris = np.asarray(tris, dtype=np.int64)
    a, b, c = pts2d[tris[:, 0]], pts2d[tris[:, 1]], pts2d[tris[:, 2]]
    flip = _cross2(b - a, c - a) < 0
    tris[flip] = tris[flip][:, ::-1]
    return tris


def _cross2(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _signed_area(poly):
    return 0.5 * np.sum(_cross2(poly, np.roll(poly, -1, axis=0)))


def _clean_polygon(poly):
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    keep = np.linalg.norm(poly - np.roll(poly, 1, axis=0), axis=1) > 1e-12
    return poly[keep]


def _point_in_triangle(p, a, b, c):
    return _cross2(b - a, p - a) >= 0 and _cross2(c - b, p - b) >= 0 and _cross2(a - c, p - c) >= 0
