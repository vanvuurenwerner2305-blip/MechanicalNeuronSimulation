"""Qt-free display meshes shared by the GUI viewport and the offscreen renders of the API: sheets shown as
solids of their thickness, stored design frames, deformed thickness."""
import numpy as np
import pyvista as pv


def polydata(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(np.asarray(vertices, dtype=float), np.hstack([np.full((len(faces), 1), 3), faces]).ravel())


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
