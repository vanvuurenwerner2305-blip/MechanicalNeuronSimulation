"""
Offscreen pictures of a design for the agent and the reports (PyVista without a window): the CAD bodies
coloured by role, cut open or exploded, and the deformed membranes coloured by displacement.

A picture holds one or more views side by side (default iso, front, top, right), each with the body names
written on it, so one image shows the whole model.
"""
import numpy as np
import pyvista as pv

from app.meshview import current_thickness, frame_polydata, polydata, solid_shell
from app.project import CHAMBER, FLUID, RIGID, ROLE_COLORS, SHEETS, part_color

from .util import ApiError

VIEWS = {  # camera direction (from the model towards the camera) and up vector
    "iso": ((1.0, -1.0, 0.8), (0, 0, 1)),
    "front": ((0, -1, 0), (0, 0, 1)),
    "back": ((0, 1, 0), (0, 0, 1)),
    "right": ((1, 0, 0), (0, 0, 1)),
    "left": ((-1, 0, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, 1, 0)),
}
SECTION_VIEW = {"x": "right", "y": "front", "z": "top"}
DEFAULT_VIEWS = ("iso", "front", "top", "right")


def parse_views(text):
    """'iso,front' / 'section-x' / 'section-y:0.3' -> [(view name, section (axis, fraction) or None)]."""
    out = []
    for item in [v.strip() for v in str(text).split(",") if v.strip()]:
        if item.startswith("section"):
            rest = item[len("section"):].lstrip("-_")
            axis, _, frac = rest.partition(":")
            if axis not in ("x", "y", "z"):
                raise ApiError(f"Unknown section {item!r}", "Use section-x, section-y or section-z (optionally :0.3 "
                                                            "for the cut position as a fraction of the model).")
            out.append((SECTION_VIEW[axis], (axis, float(frac) if frac else 0.5)))
        elif item in VIEWS:
            out.append((item, None))
        else:
            raise ApiError(f"Unknown view {item!r}", f"Views: {', '.join(VIEWS)}, section-x|y|z[:fraction].")
    return out or [(v, None) for v in DEFAULT_VIEWS]


def _layout(n):
    return (1, n) if n <= 3 else (2, (n + 1) // 2)


def _camera(plotter, bounds, view):
    direction, up = VIEWS[view]
    centre = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-9)
    plotter.camera_position = [tuple(centre + 3 * size * d), tuple(centre), up]
    plotter.reset_camera()
    plotter.camera.zoom(1.15)


def _clip(mesh, section, bounds):
    if section is None:
        return mesh
    axis, fraction = section
    k = "xyz".index(axis)
    origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    origin[k] = bounds[2 * k] + fraction * (bounds[2 * k + 1] - bounds[2 * k])
    normal = [0, 0, 0]
    normal[k] = 1.0  # keeps the half behind the cut (the camera looks at the cut face)
    try:
        return mesh.clip(normal=normal, origin=origin, invert=True)
    except Exception:
        return mesh


def _bounds(meshes):
    b = np.array([m.bounds for m in meshes if m.n_points])
    return [b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max(), b[:, 4].min(), b[:, 5].max()]


def _plot(items, path, views, title="", size=None, labels=True, scalar=None):
    """items: [{"mesh", "name", "color" | None, "opacity", "label"}]; scalar: (array name, title, clim) when some
    items are coloured by values."""
    if isinstance(views, str):
        views = parse_views(views)
    elif views and isinstance(views[0], str):
        views = parse_views(",".join(views))
    rows, cols = _layout(len(views))
    size = size or (min(1400, 470 * cols + (80 if scalar else 0)), 420 * rows)
    plotter = pv.Plotter(off_screen=True, shape=(rows, cols), window_size=size, border=False)
    bounds = _bounds([it["mesh"] for it in items])
    for k, (view, section) in enumerate(views):
        plotter.subplot(k // cols, k % cols)
        plotter.set_background("white")
        for it in items:
            mesh = _clip(it["mesh"], section, bounds)
            if mesh.n_points == 0:
                continue
            if it.get("color") is None and scalar is not None:
                plotter.add_mesh(mesh, scalars=scalar[0], cmap="turbo", clim=scalar[2], show_scalar_bar=False,
                                 opacity=it.get("opacity", 1.0), smooth_shading=False)
            else:
                plotter.add_mesh(mesh, color=it["color"], opacity=it.get("opacity", 1.0), show_edges=False,
                                 smooth_shading=False)
        if labels:
            pts, names = [], []
            for it in items:
                if it.get("label") and it["mesh"].n_points:
                    m = _clip(it["mesh"], section, bounds)
                    if m.n_points:
                        pts.append(np.asarray(m.center))
                        names.append(it["label"])
            if pts:
                plotter.add_point_labels(np.array(pts), names, font_size=10, point_size=1, shape_opacity=0.55,
                                         always_visible=True, text_color="black", shape_color="white",
                                         show_points=False)
        caption = view + (f" (cut at {section[0]} = {section[1]:.0%})" if section else "")
        plotter.add_text(caption, font_size=9, position="upper_left", color="#333333")
        plotter.add_axes(interactive=False, line_width=2, labels_off=False)
        _camera(plotter, bounds, view)
    if scalar is not None:
        plotter.subplot(0, cols - 1)
        dummy = pv.PolyData(np.zeros((2, 3)))
        dummy.point_data[scalar[0]] = np.array(scalar[2], float)
        plotter.add_mesh(dummy, scalars=scalar[0], cmap="turbo", clim=scalar[2], opacity=0.0,
                         scalar_bar_args=dict(title=scalar[1], vertical=True, position_x=0.86, position_y=0.12,
                                              height=0.7, width=0.05, title_font_size=12, label_font_size=11,
                                              fmt="%.3g", color="black"))
    if title:
        plotter.subplot(0, 0)
        plotter.add_text(title, font_size=10, position="lower_left", color="black")
    plotter.screenshot(str(path))
    plotter.close()
    return str(path)


# -----------------------------
# The model
# -----------------------------

def role_label(part):
    role = part.role
    model = part.props.get("model")
    if role in (CHAMBER, FLUID) and model:
        short = {"Constant pressure (input)": "input", "Closed: ideal gas (isothermal)": "gas",
                 "Closed: incompressible": "liquid", "Vent (open to surroundings, 0 kPa)": "vent",
                 "Constant pressure": "constant", "Dynamic pressure": "dynamic"}.get(model, model)
        return f"{part.name} ({short})"
    if role == RIGID and part.props.get("motion", "").startswith("Free"):
        return f"{part.name} (free rigid)"
    return f"{part.name} ({role.lower()})"


def render_model(surfaces, parts, path, views=DEFAULT_VIEWS, exploded=0.0, only=None, title="", labels=True):
    """The CAD bodies coloured by role (fluids see-through). exploded > 0 moves every body away from the model
    centre by that fraction of its distance; only = body names to show."""
    meshes = {i: polydata(m.vertices, m.faces) for i, m in surfaces.items()}
    if exploded:
        centre = np.array(_bounds(list(meshes.values()))).reshape(3, 2).mean(axis=1)
        for i, m in meshes.items():
            m.points = m.points + exploded * (np.asarray(m.center) - centre)
    items = []
    for i, m in meshes.items():
        part = parts[i]
        if only and part.name not in only:
            continue
        fluid = part.role in (CHAMBER, FLUID)
        items.append({"mesh": m, "name": part.name, "color": part_color(part),
                      "opacity": 0.35 if fluid else (0.3 if part.role == RIGID else 1.0),
                      "label": role_label(part)})
    if not items:
        raise ApiError("Nothing to show", "Check the body names given to --only.")
    return _plot(items, path, views, title, labels=labels)


# -----------------------------
# Results
# -----------------------------

def render_deformed(surfaces, parts, shells, coords, path, views=("iso", "front", "right"), scale=1.0, title="",
                    context=True):
    """Deformed simulated bodies (shells as solids of their thickness) coloured by |u|; rigid parts and the
    activation membranes as see-through context. shells: {body index: simulated body}; coords: {index: (n, 3)}."""
    items, values = [], []
    for i, body in shells.items():
        X = body.X.detach().cpu().numpy()
        x = np.asarray(coords[i], float)
        u = np.linalg.norm(x - X, axis=1)
        shown = X + scale * (x - X)
        faces = body.faces_np if hasattr(body, "faces_np") else body.faces.cpu().numpy()
        if hasattr(body, "thickness") and not hasattr(body, "tets") and not getattr(body, "is_rigid", False):
            pd = solid_shell(shown, faces, current_thickness(body, x), u)
        else:
            pd = polydata(shown, faces)
            pd.point_data["values"] = u
        items.append({"mesh": pd, "name": parts[i].name, "color": None, "label": parts[i].name})
        values.append(u)
    if context:
        for i, m in surfaces.items():
            if i in shells:
                continue
            part = parts[i]
            if part.role == RIGID or part.role in SHEETS:
                items.append({"mesh": polydata(m.vertices, m.faces), "name": part.name,
                              "color": ROLE_COLORS.get(part.role, "#999999"), "opacity": 0.25, "label": None})
    allv = np.concatenate(values) if values else np.zeros(1)
    clim = (float(allv.min()), float(max(allv.max(), allv.min() + 1e-12)))
    return _plot(items, path, views, title, scalar=("values", "|u| [mm]", clim))


def render_frames(frames, path, views=("iso", "front", "right"), title="", place=None):
    """Stored design bodies (DesignFrames.at(dp) or a characterisation's shapes) coloured by |u|."""
    items, values = [], []
    for f in frames:
        pd = frame_polydata(f, 1.0, place)
        items.append({"mesh": pd, "name": f["name"], "color": None, "label": f["name"]})
        values.append(np.asarray(pd.point_data["values"]))
    allv = np.concatenate(values) if values else np.zeros(1)
    clim = (float(allv.min()), float(max(allv.max(), allv.min() + 1e-12)))
    return _plot(items, path, views, title, scalar=("values", "|u| [mm]", clim))
