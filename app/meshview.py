"""Qt-free display meshes shared by the GUI viewport and the offscreen renders of the API: sheets shown as
solids of their thickness, stored design frames, deformed thickness."""
import numpy as np
import pyvista as pv


def polydata(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(np.asarray(vertices, dtype=float), np.hstack([np.full((len(faces), 1), 3), faces]).ravel())


DISPLACEMENT, STRAIN = "Displacement", "Strain"
COLOUR_FIELDS = (DISPLACEMENT, STRAIN)
COLOUR_TITLES = {DISPLACEMENT: "|u| [mm]", STRAIN: "max principal strain"}


def principal_strain(rest, x, faces):
    """Largest principal in-plane strain of each triangle, lambda_1 - 1 (engineering strain along the most stretched
    direction), from its rest and deformed corners: F maps the rest edges (in the triangle's own 2D frame) to the
    deformed ones and lambda_1^2 is the largest eigenvalue of C = F^T F. Degenerate rest triangles give 0."""
    rest, x, F = np.asarray(rest, float), np.asarray(x, float), np.asarray(faces, np.int64)
    e1, e2 = rest[F[:, 1]] - rest[F[:, 0]], rest[F[:, 2]] - rest[F[:, 0]]
    t1 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-300)
    n = np.cross(e1, e2)
    t2 = np.cross(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300), t1)
    D = np.stack([np.stack([(e1 * t1).sum(1), (e2 * t1).sum(1)], 1),
                  np.stack([(e1 * t2).sum(1), (e2 * t2).sum(1)], 1)], 1)          # (m, 2, 2) rest edges in 2D
    d = np.stack([x[F[:, 1]] - x[F[:, 0]], x[F[:, 2]] - x[F[:, 0]]], 2)           # (m, 3, 2) deformed edges
    det = D[:, 0, 0] * D[:, 1, 1] - D[:, 0, 1] * D[:, 1, 0]
    ok = np.abs(det) > 1e-12 * np.maximum((e1 ** 2).sum(1), 1e-300)
    inv = np.zeros_like(D)
    inv[ok] = np.linalg.inv(D[ok])
    Fd = d @ inv
    C = np.transpose(Fd, (0, 2, 1)) @ Fd
    half_trace = 0.5 * (C[:, 0, 0] + C[:, 1, 1])
    root = np.sqrt(np.maximum(0.25 * (C[:, 0, 0] - C[:, 1, 1]) ** 2 + C[:, 0, 1] ** 2, 0.0))
    return np.where(ok, np.sqrt(np.maximum(half_trace + root, 0.0)) - 1.0, 0.0)


def colour_values(rest, x, faces, field):
    """(values, on_cells) of a deformed body for a COLOUR_FIELDS entry: |u| per vertex or strain per triangle."""
    if field == STRAIN:
        return principal_strain(rest, x, faces), True
    return np.linalg.norm(np.asarray(x, float) - np.asarray(rest, float), axis=1), False


def solid_shell(x, faces, thickness, values=None, on_cells=False):
    """Closed solid of a deformed mid-surface: top and bottom faces half a thickness either side
    along the vertex normals, joined by side walls along the boundary. thickness is a scalar or
    per vertex. values (per vertex or per face) are carried over to the solid."""
    x, F = np.asarray(x, dtype=float), np.asarray(faces, dtype=np.int64)
    n = len(x)
    face_n = np.cross(x[F[:, 1]] - x[F[:, 0]], x[F[:, 2]] - x[F[:, 0]])  # area weighted
    normals = np.zeros_like(x)
    for k in range(3):
        np.add.at(normals, F[:, k], face_n)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
    half = 0.5 * np.broadcast_to(np.asarray(thickness, dtype=float), (n,))[:, None]
    points = np.vstack([x + half * normals, x - half * normals])

    # Boundary edges (used by one face), kept in that face's winding direction
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    owner = np.tile(np.arange(len(F)), 3)
    key = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = counts[inverse.ravel()] == 1
    a, b = edges[boundary, 0], edges[boundary, 1]
    sides = np.concatenate([np.stack([a, b, b + n], 1), np.stack([a, b + n, a + n], 1)])
    solid_faces = np.vstack([F, F[:, ::-1] + n, sides[:, ::-1]])
    pd = polydata(points, solid_faces)
    if values is not None:
        values = np.asarray(values)
        if on_cells:
            side_values = np.tile(values[owner[boundary]], 2)
            pd.cell_data["values"] = np.concatenate([values, values, side_values])
        else:
            pd.point_data["values"] = np.concatenate([values, values])
    return pd


def frame_polydata(frame, scale=1.0, place=None):
    """A stored design body (DesignFrames.at) as a mesh at its deformed coordinates, with the displacement
    magnitude as point values "values". scale exaggerates the displacement; place(points) positions it."""
    rest, x = frame["rest"], frame["x"]
    u = np.linalg.norm(x - rest, axis=1)
    shown = rest + scale * (x - rest)
    if place is not None:
        shown = place(shown)
    if frame["kind"] == "shell":
        return solid_shell(shown, frame["faces"], float(frame.get("thickness") or 0.0), u)
    pd = polydata(shown, frame["faces"])
    pd.point_data["values"] = u
    return pd


def body_kind(body):
    """'shell' (mid-surface sheet), 'solid' (tetrahedra) or 'rigid' (moving rigid body)."""
    if getattr(body, "is_rigid", False):
        return "rigid"
    if hasattr(body, "tets"):
        return "solid"
    return "shell"


def current_thickness(shell, x):
    """Per-vertex thickness of the deformed shell: incompressible rubber thins as it stretches
    (t = t0 * A0 / A with nodal areas); other materials keep the rest thickness."""
    t0 = float(shell.thickness)
    if shell.material != "neo_hookean":
        return np.full(len(x), t0)
    F = shell.faces.cpu().numpy()
    area = 0.5 * np.linalg.norm(np.cross(x[F[:, 1]] - x[F[:, 0]], x[F[:, 2]] - x[F[:, 0]]), axis=1)
    nodal = np.zeros(len(x))
    for k in range(3):
        np.add.at(nodal, F[:, k], area / 3.0)
    rest = shell.nodal_area.cpu().numpy()
    return t0 * rest / np.maximum(nodal, 1e-300)
