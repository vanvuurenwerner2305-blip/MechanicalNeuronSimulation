"""3D viewport: CAD parts coloured by role, picking, mesh view, results view, section cuts."""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from .meshview import body_kind, current_thickness, frame_polydata, polydata, solid_shell  # noqa: F401
from .project import ACTIVATION_MEMBRANE, FREE_RIGID_COLOR, RIGID, ROLE_COLORS, SHEETS, part_color, part_opacity

SELECTED = "#ffcc00"
AXES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}

RESULT_FIELDS = ["Displacement magnitude", "Displacement X", "Displacement Y", "Displacement Z", "Area stretch"]


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
        # Click selection with our own ray cast: a click (press and release without dragging)
        # picks the nearest visible part; clicking the same spot again picks the next one behind.
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)
        self._press_pos = None
        self._last_click = None      # (position, ordered hit list, index in list)

        self.show_edges = False
        self.opacity = 0.45          # opacity of parts that are not selected (transparency slider)
        self.section = None          # (axis name, fraction 0..1) or None
        self._names = []
        self._pick_meshes = {}
        self._bounds = None
        self._has_camera = False

    # -----------------------------
    # Helpers
    # -----------------------------

    def _on_press(self, *_):
        self._press_pos = np.array(self.plotter.iren.get_event_position(), dtype=float)

    def _on_release(self, *_):
        pos = np.array(self.plotter.iren.get_event_position(), dtype=float)
        if self._press_pos is None or np.abs(pos - self._press_pos).max() > 4:
            return  # that was a drag (rotate/pan), not a click
        hits = self._bodies_under(pos)
        add = bool(self.plotter.iren.interactor.GetControlKey())
        if not hits:
            self._last_click = None
            if not add:
                QTimer.singleShot(0, lambda: self.picked.emit(-1, False))
            return
        k = 0
        if self._last_click is not None:
            last_pos, last_hits, last_k = self._last_click
            if np.abs(pos - last_pos).max() <= 4 and last_hits == hits:
                k = (last_k + 1) % len(hits)
        self._last_click = (pos, hits, k)
        index = hits[k]
        # leave VTK's event handler before the scene is rebuilt
        QTimer.singleShot(0, lambda: self.picked.emit(index, add))

    def _bodies_under(self, pos):
        """Visible parts along the view ray through a display position, nearest first."""
        renderer = self.plotter.renderer
        ends = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(pos[0], pos[1], depth)
            renderer.DisplayToWorld()
            w = np.array(renderer.GetWorldPoint())
            ends.append(w[:3] / w[3])
        start, end = ends
        hits = []
        for index, mesh in self._pick_meshes.items():
            points, _ = mesh.ray_trace(start, end, first_point=True)
            points = np.asarray(points).reshape(-1, 3)
            if len(points):
                hits.append((float(np.linalg.norm(points[0] - start)), index))
        return [index for _, index in sorted(hits)]

    def _clear(self):
        for name in self._names:
            self.plotter.remove_actor(name, render=False)
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title, render=False)
        self._names, self._pick_meshes = [], {}

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
            self._pick_meshes[body] = mesh
        return actor

    def _selected_opacity(self):
        """The selected part stays see-through (so selecting an enclosure does not hide what is
        inside); its outline makes it stand out."""
        return min(1.0, self.opacity + 0.3)

    def _outline(self, index, pd):
        edges = self._clip(pd).extract_feature_edges(feature_angle=30, boundary_edges=True,
                                                     non_manifold_edges=False, manifold_edges=False)
        if edges.n_points:
            self.plotter.add_mesh(edges, name=f"outline{index}", color="#d35400", line_width=3,
                                  render=False, pickable=False)
            self._names.append(f"outline{index}")

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
            pd = polydata(mesh.vertices, mesh.faces)
            self._add(f"body{index}", pd, body=index,
                      color=SELECTED if selected else part_color(part),
                      opacity=self._selected_opacity() if selected else min(part_opacity(part), self.opacity),
                      show_edges=self.show_edges, edge_color="#40464d", line_width=0.5,
                      smooth_shading=False, pickable=True)
            if selected:
                self._outline(index, pd)
        self._finish()

    def show_mesh(self, surfaces, parts, selection, mesh_data, build=None):
        """Simulation mesh: mid-surfaces of membranes/shells (with fixed nodes), other parts as bodies."""
        self._clear()
        for index, mesh in surfaces.items():
            part = parts[index]
            if not part.visible:
                continue
            selected = index in selection
            color = SELECTED if selected else part_color(part)
            if part.role in SHEETS and index in mesh_data.midsurfaces:
                mid = mesh_data.midsurfaces[index]
                self._add(f"body{index}", polydata(mid.vertices, mid.faces), body=index, color=color,
                          opacity=1.0 if selected else max(self.opacity, 0.6), show_edges=True,
                          edge_color="#202020", line_width=0.6, pickable=True)
                if build is not None and index in build.shells:
                    fixed = build.shells[index].fixed.cpu().numpy()
                    if fixed.any():
                        self._add(f"fixed{index}", pv.PolyData(mid.vertices[fixed]), color="#1b4f9c",
                                  point_size=6, render_points_as_spheres=True, pickable=False)
                    tied = getattr(build, "tie_nodes", {}).get(index)
                    if tied is not None and len(tied):
                        self._add(f"tied{index}", pv.PolyData(mid.vertices[tied]), color=FREE_RIGID_COLOR,
                                  point_size=6, render_points_as_spheres=True, pickable=False)
            elif index in getattr(mesh_data, "volumes", {}):
                vol = mesh_data.volumes[index]
                self._add(f"body{index}", polydata(vol.vertices, vol.faces), body=index, color=color,
                          opacity=1.0 if selected else max(self.opacity, 0.6), show_edges=True,
                          edge_color="#202020", line_width=0.4, pickable=True)
                if build is not None and index in build.shells:
                    fixed = build.shells[index].fixed.cpu().numpy()
                    if fixed.any():
                        self._add(f"fixed{index}", pv.PolyData(vol.vertices[fixed]), color="#1b4f9c",
                                  point_size=5, render_points_as_spheres=True, pickable=False)
            else:
                self._add(f"body{index}", polydata(mesh.vertices, mesh.faces), body=index, color=color,
                          opacity=self._selected_opacity() if selected
                          else min(part_opacity(part), self.opacity, 0.35),
                          show_edges=True, edge_color="#40464d", line_width=0.4, pickable=True)
        self._finish()

    def show_results(self, surfaces, parts, build, step, field, scale=1.0, clim=None):
        """Deformed membranes/shells coloured by a result field; rigid parts as context."""
        self._clear()
        for index, mesh in surfaces.items():
            if parts[index].role == RIGID and parts[index].visible:
                self._add(f"body{index}", polydata(mesh.vertices, mesh.faces), body=index,
                          color=ROLE_COLORS[RIGID], opacity=min(0.3, self.opacity), pickable=True)
            elif parts[index].role == ACTIVATION_MEMBRANE and parts[index].visible:  # not simulated: as in CAD
                self._add(f"body{index}", polydata(mesh.vertices, mesh.faces), body=index,
                          color=ROLE_COLORS[ACTIVATION_MEMBRANE], opacity=min(0.6, self.opacity), pickable=True)

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
            data, on_cells = values[index]
            kind = body_kind(shell)
            if kind == "shell":
                # The whole membrane body (thickness from the true deformed state, not the scaled one)
                pd = solid_shell(x, shell.faces.cpu().numpy(), current_thickness(shell, coords[index].numpy()),
                                 data, on_cells)
            else:
                pd = polydata(x, shell.faces.cpu().numpy())
                if on_cells:
                    pd.cell_data["values"] = data
                else:
                    pd.point_data["values"] = data
            pd.rename_array("values", field)
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
                if body_kind(shell) == "rigid":
                    out[index] = (np.ones(len(F)), True)
                    continue
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
