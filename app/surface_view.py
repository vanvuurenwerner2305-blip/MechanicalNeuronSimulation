"""
Surfaces z(x, y) drawn by VTK (on the graphics card, so turning them is smooth), one per subplot of a grid that
shares one camera. Used by the Analysis tab.

Every panel is drawn in the same unit box (x, y over 0..1, z over 0..HEIGHT) with its axes labelled in the real
values (`axes_ranges`), so parameters with different units look alike. One colour scale for all panels.
"""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import QEvent, QTimer, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

HEIGHT = 0.7            # height of the z range in the unit box
VIEW_DISTANCE = 3.6       # camera distance from the box centre (box diagonal about 1.6)
PICK_PIXELS = 12        # a click picks the nearest point within this distance (display pixels)
COLORS = {"point": "#202020", "extrapolated": "#c62828", "unconverged": "#000000", "selected": "#ff6f00"}


def _surface(xs, ys, z, lo, hi):
    """The quads of the grid whose four corners are solved, in the unit box, with z as the scalars."""
    nx, ny = len(xs), len(ys)
    u = (xs - xs[0]) / (xs[-1] - xs[0]) if nx > 1 and xs[-1] != xs[0] else np.zeros(nx)
    v = (ys - ys[0]) / (ys[-1] - ys[0]) if ny > 1 and ys[-1] != ys[0] else np.zeros(ny)
    U, V = np.meshgrid(u, v, indexing="ij")
    W = HEIGHT * (z - lo) / (hi - lo)
    points = np.column_stack([U.ravel(), V.ravel(), np.nan_to_num(W).ravel()])
    ok = np.isfinite(z)
    quads = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if ok[i, j] and ok[i + 1, j] and ok[i + 1, j + 1] and ok[i, j + 1]:
                quads.append([4, i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1])
    mesh = pv.PolyData(points, np.asarray(quads, np.int64).ravel()) if quads else None
    if mesh is not None:
        mesh.point_data["z"] = np.nan_to_num(z).ravel()
    return mesh, points.reshape(nx, ny, 3)


class SurfaceGrid(QWidget):
    picked = Signal(int, int, int, int)   # panel row, column, x index, y index
    scrolled = Signal(int)                # +1 / -1 (only while scroll_layers is set; otherwise the wheel zooms)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        self.shape = None
        self.scroll_layers = False
        self._points = {}            # (row, col) -> unit-box points (nx, ny, 3)
        self._press = None

    def _make(self, shape):
        if self.plotter is not None and self.shape == shape:
            return False
        if self.plotter is not None:
            old = self.plotter
            self.layout_.removeWidget(old.interactor)
            old.close()
            old.interactor.deleteLater()
        self.plotter = QtInteractor(self, shape=shape, border=False)
        self.layout_.addWidget(self.plotter.interactor)
        self.plotter.set_background("#ffffff")
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)
        self.plotter.interactor.installEventFilter(self)
        self.shape = shape
        return True

    def eventFilter(self, obj, event):
        if self.scroll_layers and event.type() == QEvent.Wheel:
            self.scrolled.emit(1 if event.angleDelta().y() > 0 else -1)
            return True        # the layers change instead of zooming
        return super().eventFilter(obj, event)

    def draw(self, panels, xs, ys, zlim, labels, selected=None, scalar_title="z"):
        """panels: [[{"z", "extrapolated", "unconverged", "title"}]] (rows of columns); labels: (x, y, z) titles;
        selected: (row, col, i, j) of the point to ring, or None."""
        shape = (len(panels), len(panels[0]))
        new = self._make(shape)
        p = self.plotter
        p.clear()
        lo, hi = zlim
        self._points = {}
        ranges = [xs[0], xs[-1], ys[0], ys[-1], lo, hi]
        for r, row in enumerate(panels):
            for c, panel in enumerate(row):
                p.subplot(r, c)
                mesh, pts = _surface(xs, ys, panel["z"], lo, hi)
                self._points[r, c] = (pts, np.isfinite(panel["z"]))
                last = r == 0 and c == shape[1] - 1
                if mesh is not None:
                    p.add_mesh(mesh, scalars="z", cmap="viridis", clim=(lo, hi), show_edges=True,
                               edge_color="#333333", line_width=0.5, show_scalar_bar=last,
                               scalar_bar_args=dict(title=scalar_title, vertical=True, position_x=0.94,
                                                    position_y=0.1, height=0.75, width=0.022, fmt="%.3g",
                                                    color="black", title_font_size=11, label_font_size=10))
                ok = np.isfinite(panel["z"])
                for key, mask, size in (("point", ok, 6), ("extrapolated", ok & panel["extrapolated"], 11),
                                        ("unconverged", ok & panel["unconverged"], 13)):
                    if mask.any():
                        p.add_points(pts[mask], color=COLORS[key], point_size=size, render_points_as_spheres=True,
                                     pickable=False)
                if selected is not None and selected[:2] == (r, c):
                    p.add_points(pts[selected[2], selected[3]][None, :], color=COLORS["selected"], point_size=20,
                                 render_points_as_spheres=True, pickable=False)
                if panel.get("title"):
                    p.add_text(panel["title"], position="upper_left", font_size=8, color="black")
                # last: adding an actor afterwards resets the labelled ranges to the unit box's 0..1
                box = pv.Box(bounds=(0, 1, 0, 1, 0, HEIGHT))
                p.show_bounds(mesh=box, axes_ranges=ranges, xtitle=labels[0], ytitle=labels[1], ztitle=labels[2],
                              font_size=8, color="#333333", grid=False, location="outer", fmt="%.3g",
                              n_xlabels=min(len(xs), 5), n_ylabels=min(len(ys), 5), n_zlabels=4)
        if new:
            p.link_views()
            self.reset_view()
        p.render()

    def reset_view(self):
        # the same view of the unit box for every panel (one shared camera), set explicitly: reset_camera before
        # the widget has its final size zoomed in too far
        if self.plotter is None:
            return
        self.plotter.subplot(0, 0)
        focal = np.array([0.5, 0.5, 0.5 * HEIGHT])
        direction = np.array([1.3, -1.7, 1.1])
        self.plotter.camera_position = [tuple(focal + VIEW_DISTANCE * direction / np.linalg.norm(direction)),
                                        tuple(focal), (0.0, 0.0, 1.0)]
        self.plotter.render()

    # -----------------------------
    # Clicking a point
    # -----------------------------

    def _on_press(self, *_):
        self._press = np.array(self.plotter.iren.get_event_position(), float)

    def _on_release(self, *_):
        pos = np.array(self.plotter.iren.get_event_position(), float)
        if self._press is None or np.abs(pos - self._press).max() > 4:
            return     # a drag (turning), not a click
        renderer = self.plotter.iren.interactor.FindPokedRenderer(int(pos[0]), int(pos[1]))
        renderers = list(self.plotter.renderers)
        if renderer not in renderers:
            return
        loc = self.plotter.renderers.index_to_loc(renderers.index(renderer))
        r, c = (int(loc[0]), int(loc[1])) if np.ndim(loc) else (0, int(loc))
        if (r, c) not in self._points:
            return
        pts, ok = self._points[r, c]
        best, where = PICK_PIXELS * self.devicePixelRatioF(), None
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                if not ok[i, j]:
                    continue
                renderer.SetWorldPoint(*pts[i, j], 1.0)
                renderer.WorldToDisplay()
                d = np.hypot(*(np.array(renderer.GetDisplayPoint()[:2]) - pos))
                if d < best:
                    best, where = d, (i, j)
        if where is not None:   # leave VTK's handler before the plots are rebuilt
            QTimer.singleShot(0, lambda: self.picked.emit(r, c, *where))

    def screenshot(self, path):
        if self.plotter is not None:
            self.plotter.screenshot(str(path))

    def close(self):
        if self.plotter is not None:
            self.plotter.close()
