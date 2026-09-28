"""3D viewport: CAD parts coloured by role, picking, mesh view, results view, section cuts."""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

from .project import DEFORMABLE, RIGID, ROLE_COLORS, ROLE_OPACITY

SELECTED = "#ffcc00"
AXES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}

RESULT_FIELDS = ["Displacement magnitude", "Displacement X", "Displacement Y", "Displacement Z", "Area stretch"]


def polydata(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(np.asarray(vertices, dtype=float), np.hstack([np.full((len(faces), 1), 3), faces]).ravel())


class Viewport(QWidget):
    picked = Signal(int, bool)  # body index, add to selection (Ctrl held)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = QtInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)

        self.plotter.set_background("#dfe5ec", top="#ffffff")
        self.plotter.add_axes(interactive=False)
        self.plotter.enable_depth_peeling()
        self.plotter.enable_mesh_picking(self._on_pick, use_actor=True, show=False, show_message=False,
                                         left_clicking=True)

        self.show_edges = False
        self.section = None          # (axis name, fraction 0..1) or None
        self._names = []
        self._actor_body = {}
        self._bounds = None
        self._has_camera = False

    # -----------------------------
    # Helpers
    # -----------------------------

    def _on_pick(self, actor):
        index = self._actor_body.get(id(actor))
        if index is not None:
            add = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
            self.picked.emit(index, add)

    def _clear(self):
        for name in self._names:
            self.plotter.remove_actor(name, render=False)
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title, render=False)
        self._names, self._actor_body = [], {}

    def _clip(self, mesh):
        if self.section is None or self._bounds is None:
            return mesh
        axis, fraction = self.section
        k = "XYZ".index(axis)
        origin = self._bounds[:3] + 0.5 * (self._bounds[3:] - self._bounds[:3])
        origin[k] = self._bounds[k] + fraction * (self._bounds[3 + k] - self._bounds[k])
        return mesh.clip(normal=AXES[axis], origin=origin, invert=True)

    def _add(self, name, mesh, body=None, **kwargs):
        mesh = self._clip(mesh)
        if mesh.n_points == 0:
            return None
        actor = self.plotter.add_mesh(mesh, name=name, render=False, **kwargs)
        self._names.append(name)
        if body is not None:
            self._actor_body[id(actor)] = body
        return actor

    def set_bounds(self, meshes):
        pts = np.vstack([m.vertices for m in meshes.values()])
        self._bounds = np.concatenate([pts.min(0), pts.max(0)])

    def _finish(self):
        if not self._has_camera:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self._has_camera = True
        self.plotter.render()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def new_model(self):
        self._has_camera = False

    # -----------------------------
    # Scenes
    # -----------------------------

    def show_model(self, surfaces, parts, selection):
        """CAD bodies (their surface meshes) coloured by role."""
        self._clear()
        for index, mesh in surfaces.items():
            part = parts[index]
            if not part.visible:
                continue
            selected = index in selection
            self._add(f"body{index}", polydata(mesh.vertices, mesh.faces), body=index,
                      color=SELECTED if selected else ROLE_COLORS[part.role],
                      opacity=1.0 if selected else ROLE_OPACITY[part.role],
                      show_edges=self.show_edges, edge_color="#40464d", line_width=0.5,
                      smooth_shading=False, pickable=True)
        self._finish()

    def show_mesh(self, surfaces, parts, selection, mesh_data, build=None):
        """Simulation mesh: mid-surfaces of membranes/shells (with fixed nodes), other parts as bodies."""
        self._clear()
        for index, mesh in surfaces.items():
            part = parts[index]
            if not part.visible:
                continue
            selected = index in selection
            color = SELECTED if selected else ROLE_COLORS[part.role]
            if part.role in DEFORMABLE and index in mesh_data.midsurfaces:
                mid = mesh_data.midsurfaces[index]
                self._add(f"body{index}", polydata(mid.vertices, mid.faces), body=index, color=color,
                          show_edges=True, edge_color="#202020", line_width=0.6, pickable=True)
                if build is not None and index in build.shells:
                    fixed = build.shells[index].fixed.cpu().numpy()
                    if fixed.any():
                        self._add(f"fixed{index}", pv.PolyData(mid.vertices[fixed]), color="#1b4f9c",
                                  point_size=6, render_points_as_spheres=True, pickable=False)
            else:
                self._add(f"body{index}", polydata(mesh.vertices, mesh.faces), body=index, color=color,
                          opacity=1.0 if selected else min(ROLE_OPACITY[part.role], 0.35),
                          show_edges=True, edge_color="#40464d", line_width=0.4, pickable=True)
        self._finish()

    def show_results(self, surfaces, parts, build, step, field, scale=1.0, clim=None):
        """Deformed membranes/shells coloured by a result field; rigid parts as context."""
        self._clear()
        for index, mesh in surfaces.items():
            if parts[index].role == RIGID and parts[index].visible:
                self._add(f"body{index}", polydata(mesh.vertices, mesh.faces), body=index,
                          color=ROLE_COLORS[RIGID], opacity=0.3, pickable=True)

        values = self.result_values(build, step, field)
        if clim is None:
            all_values = np.concatenate([v for v, _ in values.values()]) if values else np.zeros(1)
            clim = (float(all_values.min()), float(all_values.max()))
            if clim[0] == clim[1]:
                clim = (clim[0], clim[0] + 1e-12)
        coords = dict(zip(build.shells.keys(), step["shell_coords"]))
        first = True
        for index, shell in build.shells.items():
            if not parts[index].visible:
                continue
            X = shell.X.cpu().numpy()
            x = X + scale * (coords[index].numpy() - X)
            pd = polydata(x, shell.faces.cpu().numpy())
            data, on_cells = values[index]
            if on_cells:
                pd.cell_data[field] = data
            else:
                pd.point_data[field] = data
            self._add(f"result{index}", pd, body=index, scalars=field, cmap="turbo", clim=clim,
                      show_edges=self.show_edges, edge_color="#303030", line_width=0.4,
                      show_scalar_bar=first, scalar_bar_args=dict(title=_field_title(field), vertical=True,
                                                                  position_x=0.86, position_y=0.1, height=0.75,
                                                                  width=0.06, title_font_size=14,
                                                                  label_font_size=12, fmt="%.3g", color="black"),
                      pickable=True)
            first = False
        self._finish()
        return clim

    @staticmethod
    def result_values(build, step, field):
        """{body index: (values, on_cells)} for one load step."""
        out = {}
        for (index, shell), coords in zip(build.shells.items(), step["shell_coords"]):
            X = shell.X.cpu().numpy()
            u = coords.numpy() - X
            if field == "Area stretch":
                F = shell.faces.cpu().numpy()
                x = coords.numpy()
                area = 0.5 * np.linalg.norm(np.cross(x[F[:, 1]] - x[F[:, 0]], x[F[:, 2]] - x[F[:, 0]]), axis=1)
                out[index] = (area / shell.rest_area.cpu().numpy(), True)
            elif field == "Displacement magnitude":
                out[index] = (np.linalg.norm(u, axis=1), False)
            else:
                out[index] = (u[:, "XYZ".index(field[-1])], False)
        return out

    def screenshot(self, path):
        self.plotter.screenshot(str(path))

    def close(self):
        self.plotter.close()


def _field_title(field):
    return {"Displacement magnitude": "|u| [mm]", "Displacement X": "u_x [mm]", "Displacement Y": "u_y [mm]",
            "Displacement Z": "u_z [mm]"}.get(field, field)
