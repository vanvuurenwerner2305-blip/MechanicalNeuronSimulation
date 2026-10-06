"""
Analysis tab: look at a full neuron's stored characterisation (a *.mfn after "Run characterisation"). Nothing is
simulated or changed here.

Choose the result to plot (Z) and tick swept parameters as plot axes in turn: the first gives a curve, the second a
surface, the third a row of surfaces, the fourth a grid and the fifth layers of grids (scroll wheel over the plots
to go up and down). A ticked parameter beyond the second shows a number of its values (how many) instead of a
slider; every unticked one is held at its slider's value. Values are either the simulated points only or
interpolated between them (multilinear). Click a point in a plot to show the neuron there in 3D.
"""
from pathlib import Path

import numpy as np
from matplotlib import cm, colors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                            QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMenu, QMessageBox, QPlainTextEdit,
                            QScrollArea, QSlider, QSpinBox, QStackedWidget, QStyle, QTabWidget, QToolBar, QTreeWidget,
                            QVBoxLayout, QWidget)

try:
    from qtpy.QtWidgets import QAction
except ImportError:  # Qt6
    from qtpy.QtGui import QAction

from .analysis import COUNTED, ROLES, evaluate_point, panels, sorted_values
from .full_neuron import ACTIVATION, MISSING, NEURON, OUTDATED, FullNeuronProject
from .full_neuron_window import FILTER, GROUP_NAMES, MODEL, RESULTS, STATUS_COLORS, FullNeuronWindow
from .panels import fmt
from .surface_view import SurfaceGrid
from .viewport import Viewport

TITLE = "Analysis"
SURFACE, MAP = "Surface", "Colour map"
CMAP = "viridis"
ROLE_NAMES = {"x": "x axis", "y": "y axis", "columns": "columns", "rows": "rows", "layers": "layers"}
SLIDER_STEPS = 1000      # slider positions when interpolating
MAX_COUNT = 12           # values of a counted parameter when interpolating

GUIDE = """
<h3>Analysis</h3>
<p>Look at a full neuron's stored characterisation without simulating anything.</p>
<ol>
<li><b>File → Open</b> a full neuron (*.mfn) that has a characterisation (Full neuron tab → Run characterisation),
or press <b>Analyse</b> in the Full neuron tab.</li>
<li>Choose the result to plot (<b>Z</b>).</li>
<li>Tick swept parameters as plot axes, in order: the 1st gives a curve, the 2nd a surface, the 3rd a row of
surfaces, the 4th a grid of surfaces and the 5th layers of grids (<b>scroll wheel</b> over the plots to go up and
down). From the 3rd on, a parameter shows <i>how many</i> of its values to use instead of a slider.</li>
<li>Unticked parameters are held at their slider's value.</li>
<li><b>Interpolate</b> off: only simulated points are used (sliders snap to them). On: values in between are
interpolated linearly from the neighbouring points.</li>
<li>Click a point in a plot to show the neuron (and the design) deformed there in the 3D view. Red points were
extrapolated beyond the design's range, black crosses did not converge, gaps were not solved.</li>
<li>Click a part in the model tree to see its settings (read only: change them in their own tabs).</li>
</ol>
"""


class AnalysisWindow(FullNeuronWindow):
    TITLE = TITLE

    def __init__(self):
        self.dataset = None
        self.shapes = None
        self.plot_axes = []        # axis indices in the order they were ticked (roles ROLES)
        self.fixed = {}            # axis -> slider value
        self.counts = {}           # axis -> number of values (counted roles)
        self.point = {}            # axis -> value of the point shown in 3D
        self.layer = 0
        self.axis_rows = []        # per axis: widgets
        self._picks = {}           # matplotlib artist -> (panel "at", x values, y values or None)
        self._data = None          # the drawn panels (analysis.panels)
        super().__init__()
        self.log.clear()
        self.log_message("Analysis: File → Open a full neuron (*.mfn) with a stored characterisation, or press "
                         "Analyse in the Full neuron tab. Help → Quick guide explains the rest.")

    # -----------------------------
    # UI
    # -----------------------------

    def _build_ui(self):
        self.figure = Figure(figsize=(7, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.surfaces_view = SurfaceGrid()      # surfaces on the graphics card (smooth turning)
        self.surfaces_view.picked.connect(self._on_surface_pick)
        self.surfaces_view.scrolled.connect(self._change_layer)
        self.plots = QStackedWidget()
        self.plots.addWidget(self.canvas)
        self.plots.addWidget(self.surfaces_view)
        self.setCentralWidget(self.plots)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Part", "Role"])
        self.tree.setColumnWidth(0, 190)
        self.tree.itemSelectionChanged.connect(self._tree_selected)
        self.tree.itemChanged.connect(self._tree_checked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        dock = QDockWidget("Model", self)
        dock.setObjectName("analysis_model_dock")
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.tree_dock = dock

        self.viewport = Viewport(self)
        self.viewport.picked.connect(self._on_pick)
        dock = QDockWidget("3D view", self)
        dock.setObjectName("analysis_view_dock")
        dock.setWidget(self.viewport)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.view_dock = dock
        self._sized = False

        self.tabs = QTabWidget()
        self.tabs.addTab(self._make_plot_tab(), "Plot")
        self.props_area = QScrollArea()
        self.props_area.setWidgetResizable(True)
        self.tabs.addTab(self.props_area, "Properties")
        dock = QDockWidget("Analysis", self)
        dock.setObjectName("analysis_controls_dock")
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(420)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        dock = QDockWidget("Messages", self)
        dock.setObjectName("analysis_log_dock")
        dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [110], Qt.Vertical)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self._show_properties()

    def _make_plot_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()
        self.z_combo = QComboBox()
        self.z_combo.currentIndexChanged.connect(lambda _: self.redraw())
        form.addRow("Z (result)", self.z_combo)
        self.style_combo = QComboBox()
        self.style_combo.addItems([SURFACE, MAP])
        self.style_combo.currentIndexChanged.connect(lambda _: self.redraw())
        form.addRow("Surfaces as", self.style_combo)
        self.interpolate = QCheckBox("Interpolate between the simulated points")
        self.interpolate.setToolTip("Off: only simulated points are used and the sliders snap to them. On: values "
                                    "in between are interpolated linearly from the neighbouring points.")
        self.interpolate.toggled.connect(self._on_interpolate)
        form.addRow(self.interpolate)
        layout.addLayout(form)

        box = QGroupBox("Swept parameters (tick in order: x, y, columns, rows, layers)")
        self.axes_grid = QGridLayout(box)
        layout.addWidget(box)

        self.layer_label = QLabel()
        self.layer_label.setWordWrap(True)
        layout.addWidget(self.layer_label)

        box = QGroupBox("Point shown in 3D (click a point in a plot)")
        v = QVBoxLayout(box)
        self.point_label = QLabel("-")
        self.point_label.setWordWrap(True)
        self.point_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.point_label)
        form = QFormLayout()
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.0, 1000.0)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.25)
        self.scale.valueChanged.connect(lambda _: self.refresh_view())
        form.addRow("Deformation scale", self.scale)
        v.addLayout(form)
        layout.addWidget(box)

        self.data_label = QLabel()
        self.data_label.setWordWrap(True)
        self.data_label.setStyleSheet("color: #555;")
        layout.addWidget(self.data_label)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        return scroll

    def _build_actions(self):
        S = QStyle
        self.a_open = self._action("Open full neuron…", self.open_project, "Ctrl+O", S.SP_DialogOpenButton,
                                   "Open a full neuron (*.mfn) with a stored characterisation")
        self.a_reset_cam = self._action("Reset camera", self.viewport.reset_camera, "R")
        self.a_guide = self._action("Quick guide", self.show_guide, "F1")
        self.view_actions = {}
        for mode, key in ((MODEL, "1"), (RESULTS, "2")):
            a = QAction(mode, self, checkable=True)
            a.setShortcut(key)
            a.triggered.connect(lambda _=False, m=mode: self.set_mode(m))
            self.view_actions[mode] = a
        self.view_actions[MODEL].setChecked(True)
        menu = self.menuBar()
        m = menu.addMenu("&File")
        m.addAction(self.a_open)
        m = menu.addMenu("&View")
        for a in (*self.view_actions.values(), self.a_reset_cam):
            m.addAction(a)
        m = menu.addMenu("&Help")
        m.addAction(self.a_guide)
        tb = QToolBar("Main")
        tb.setObjectName("analysis_toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tb.addAction(self.a_open)
        tb.addSeparator()
        for a in self.view_actions.values():
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" Transparency: "))
        self.transparency = QSlider(Qt.Horizontal)
        self.transparency.setRange(0, 95)
        self.transparency.setValue(55)
        self.transparency.setFixedWidth(120)
        self.transparency.valueChanged.connect(self._on_transparency)
        tb.addWidget(self.transparency)
        self.addToolBar(tb)

    def _update_actions(self):
        self.view_actions[RESULTS].setEnabled(self._can_show())
        name = Path(self.project.path).name if self.project.path else ""
        self.setWindowTitle(f"{TITLE} — {name}" if name else TITLE)
        text = []
        if self.dataset is not None:
            text.append(f"{name}: {int(self.dataset.solved.sum())}/{self.dataset.size} points, "
                        f"{len(self.dataset.axes)} swept parameter(s)")
            if not self.project.dataset_current():
                text.append("the model changed since the characterisation was run")
        text += [f"{self._source_name(k)} {v}" for k, v in self.statuses.items() if v in (OUTDATED, MISSING)]
        self.status_label.setText(" · ".join(text))

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:   # the 3D view gets most of the left column (docks only size once shown)
            self._sized = True
            self.resizeDocks([self.tree_dock, self.view_dock], [440, 440], Qt.Horizontal)
            self.resizeDocks([self.tree_dock, self.view_dock], [220, 800], Qt.Vertical)

    def show_guide(self):
        QMessageBox.information(self, "Quick guide - analysis", GUIDE)

    # -----------------------------
    # Files
    # -----------------------------

    def open_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open full neuron", self._dialog_dir(), FILTER)
            if not path:
                return
        self._remember_dir(path)
        try:
            project = FullNeuronProject.load(path)
            if not project.characterisation:
                raise ValueError("It has no stored characterisation: run one in the Full neuron tab first.")
            self.project = project
            self._load_neuron_cad()
            self.surfaces[ACTIVATION], self.design_project = {}, None
            if project.design_data is not None:
                self._load_design_cad()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, TITLE, f"Could not open {Path(path).name}:\n{exc}")
            return
        self.log_message(f"Opened {Path(path).name}.")
        self._after_import()

    def _after_import(self):
        self.statuses = {k: self.project.reference_status(k) for k in (NEURON, ACTIVATION)}
        self.dataset = self.project.dataset()
        self.shapes = self.dataset.shapes() if self.dataset is not None else None
        self.bodies = self.shapes or []
        self.axes_info = [(a["label"], a["unit"]) for a in self.dataset.axes] if self.dataset else []
        self.showing = "dataset"
        self.selection = None
        self.viewport.new_model()
        self._populate_tree()
        d = self.dataset
        self.plot_axes, self.layer = [], 0
        self.fixed = {k: float(sorted_values(a["values"])[0]) for k, a in enumerate(d.axes)}
        self.counts = {k: min(len(a["values"]), 3) for k, a in enumerate(d.axes)}
        self.point = dict(self.fixed)
        live = [k for k, a in enumerate(d.axes) if len(a["values"]) > 1]
        self.plot_axes = live[:min(2, len(live))]
        self.z_combo.blockSignals(True)
        self.z_combo.clear()
        for key in d.keys:
            label, unit = d.labels[key]
            self.z_combo.addItem(f"{label} [{unit}]" if unit else label, key)
        closed = next((i for i, k in enumerate(d.keys) if k.startswith("P:")), 0)
        self.z_combo.setCurrentIndex(closed)
        self.z_combo.blockSignals(False)
        notes = []
        if not d.complete:
            notes.append("The characterisation is incomplete (it was stopped): gaps are points not solved.")
        if not self.project.dataset_current():
            notes.append("The model was changed after this characterisation was run (the plots show the model as it "
                         "was then).")
        if self.shapes is None:
            notes.append("The deformed shapes were not stored: the 3D view shows the model undeformed.")
        self.data_label.setText(" ".join(notes))
        for note in notes:
            self.log_message(note)
        self._build_axes()
        self._update_point()
        self.set_mode(RESULTS)
        self.redraw()
        self._update_actions()

    def _announce_statuses(self, old=None):
        for key, status in self.statuses.items():
            if old is not None and old.get(key) == status:
                continue
            if status == OUTDATED:
                self.log_message(f"{self._source_name(key)} was saved with changes since this full neuron took it: "
                                 "update it in the Full neuron tab and run the characterisation again.")

    # -----------------------------
    # Model tree and properties (read only)
    # -----------------------------

    def _tree_selected(self):
        items = self.tree.selectedItems()
        self.selection = items[0].data(0, Qt.UserRole) if items else None
        self._show_properties()
        if self.selection is not None:
            self.tabs.setCurrentWidget(self.props_area)
        self.refresh_view()

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        data = item.data(0, Qt.UserRole) if item is not None else None
        if not data:
            return
        key = data[1] if data[0] == "group" else (NEURON if data[0] == "neuron" else ACTIVATION)
        source = self.project.neuron_source if key == NEURON else self.project.design_path
        if not source:
            return
        menu = QMenu(self)
        edit = menu.addAction(f"Open {Path(source).name} in its tab")
        edit.setEnabled(self.statuses.get(key) != MISSING)
        edit.triggered.connect(lambda: self.open_source.emit(source))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _build_properties(self):
        old = self.props_area.takeWidget()
        if old is not None:
            old.deleteLater()
        w = QWidget()
        layout = QVBoxLayout(w)
        title = QLabel()
        title.setStyleSheet("font-weight: 600; font-size: 13px;")
        title.setWordWrap(True)
        layout.addWidget(title)
        sel = self.selection
        if sel is None:
            title.setText("Click a part in the model tree to see its settings.")
        elif sel[0] == "group":
            key = sel[1]
            title.setText(GROUP_NAMES[key])
            status = self.statuses.get(key)
            source = self.project.neuron_source if key == NEURON else self.project.design_path
            info = QLabel(f"{source or ''}<br><b style='color:{STATUS_COLORS.get(status, '#2e7d32')}'>"
                          + {OUTDATED: "The file was saved with changes since (update in the Full neuron tab)",
                             MISSING: "The file was not found"}.get(status, "Up to date") + "</b>")
            info.setWordWrap(True)
            layout.addWidget(info)
            if key == ACTIVATION and self.project.design_data is not None:
                d = self.project.design()
                layout.addWidget(QLabel(f"Δp {d.dp_range[0]:.4g} … {d.dp_range[1]:.4g} kPa, outputs "
                                        f"{', '.join(d.output_names)}, membrane {d.membrane_area:.4g} mm²"))
        else:
            kind, i = sel
            part = self._part(kind, i)
            title.setText(part.name)
            linked = kind == "neuron" and part.name == self.project.link.get("part")
            layout.addWidget(QLabel(f"Role: <b>{part.role}</b>" + (" (replaced by the design's membrane)"
                                                                    if linked else "")))
            owner = self.design_project if kind == "design" else self.project.neuron
            fields = owner.role_fields.get(part.role, [])
            swept = {a["field"]: a for a in self.dataset.axes if a["part"] == part.name} if self.dataset else {}
            form = QFormLayout()
            for f in fields:
                if f.key not in part.props or (f.visible_if and part.props.get(f.visible_if[0]) not in
                                                f.visible_if[1]):
                    continue
                value = part.props[f.key]
                text = fmt(value) if isinstance(value, float) else str(value)
                if isinstance(value, list):
                    text = ", ".join(str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in value)
                if f.key in swept:
                    v = sorted_values(swept[f.key]["values"])
                    text = (f"<b>swept {fmt(v[0])} … {fmt(v[-1])}, {len(v)} points</b> (shown at "
                            f"{fmt(self.point.get(self._axis_of(part.name, f.key), v[0]))})")
                label = QLabel(f"{text} {f.unit}".strip())
                label.setWordWrap(True)
                form.addRow(f.label, label)
            layout.addLayout(form)
            note = QLabel("Read only: change it in its own tab.")
            note.setStyleSheet("color: #555;")
            layout.addWidget(note)
        layout.addStretch(1)
        self.props_area.setWidget(w)

    def _axis_of(self, part, field):
        return next((k for k, a in enumerate(self.dataset.axes) if a["part"] == part and a["field"] == field), None)

    # -----------------------------
    # Swept parameters
    # -----------------------------

    def _build_axes(self):
        while self.axes_grid.count():
            item = self.axes_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.axis_rows = []
        for c, text in enumerate(("Plot", "Parameter", "", "Value / how many")):
            label = QLabel(f"<b>{text}</b>")
            self.axes_grid.addWidget(label, 0, c)
        for k, a in enumerate(self.dataset.axes):
            n = len(a["values"])
            check = QCheckBox()
            check.setEnabled(n > 1)
            check.setToolTip("Plot against this parameter" if n > 1 else "Only one value was simulated")
            check.toggled.connect(lambda on, k=k: self._on_tick(k, on))
            name = QLabel(f"{a['label']}" + (f" [{a['unit']}]" if a["unit"] else ""))
            name.setWordWrap(True)
            role = QLabel()
            role.setStyleSheet("color: #1565c0;")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimumWidth(130)
            slider.valueChanged.connect(lambda pos, k=k: self._on_slider(k, pos))
            value = QLabel()
            value.setMinimumWidth(60)
            count = QSpinBox()
            count.setRange(1, n)
            count.setValue(self.counts[k])
            count.setSuffix(" values")
            count.valueChanged.connect(lambda v, k=k: self._on_count(k, v))
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(slider, 1)
            h.addWidget(value)
            h.addWidget(count)
            r = k + 1
            self.axes_grid.addWidget(check, r, 0)
            self.axes_grid.addWidget(name, r, 1)
            self.axes_grid.addWidget(role, r, 2)
            self.axes_grid.addWidget(holder, r, 3)
            self.axis_rows.append({"check": check, "role": role, "slider": slider, "value": value, "count": count})
        self.axes_grid.setColumnStretch(3, 1)
        self._sync_axes()

    def _role(self, k):
        return ROLES[self.plot_axes.index(k)] if k in self.plot_axes else None

    def _sync_axes(self):
        """Ticks, roles, slider ranges/positions and counts from the state."""
        interp = self.interpolate.isChecked()
        for k, row in enumerate(self.axis_rows):
            values = sorted_values(self.dataset.axes[k]["values"])
            role = self._role(k)
            for w in row.values():
                w.blockSignals(True)
            row["check"].setChecked(role is not None)
            row["check"].setEnabled(len(values) > 1 and (role is not None or len(self.plot_axes) < len(ROLES)))
            row["role"].setText(ROLE_NAMES.get(role, ""))
            slider = row["slider"]
            if interp:
                slider.setRange(0, SLIDER_STEPS)
                span = values[-1] - values[0]
                slider.setValue(int(round(SLIDER_STEPS * (self.fixed[k] - values[0]) / span)) if span else 0)
            else:
                slider.setRange(0, len(values) - 1)
                slider.setValue(int(np.argmin(np.abs(values - self.fixed[k]))))
            slider.setVisible(role is None)
            row["value"].setVisible(role is None)
            row["value"].setText(fmt(self.fixed[k]))
            row["count"].setVisible(role in COUNTED)
            row["count"].setMaximum(MAX_COUNT if interp else len(values))
            row["count"].setValue(self.counts[k])
            for w in row.values():
                w.blockSignals(False)
        layers = self.plot_axes[4] if len(self.plot_axes) > 4 else None
        self.layer_label.setVisible(layers is not None)

    def _on_tick(self, k, on):
        if on and k not in self.plot_axes and len(self.plot_axes) < len(ROLES):
            self.plot_axes.append(k)
        elif not on and k in self.plot_axes:
            self.plot_axes.remove(k)
        self.layer = 0
        self._sync_axes()
        self.redraw()

    def _on_slider(self, k, pos):
        values = sorted_values(self.dataset.axes[k]["values"])
        if self.interpolate.isChecked():
            self.fixed[k] = float(values[0] + (values[-1] - values[0]) * pos / SLIDER_STEPS)
        else:
            self.fixed[k] = float(values[min(pos, len(values) - 1)])
        self.axis_rows[k]["value"].setText(fmt(self.fixed[k]))
        self.point[k] = self.fixed[k]
        self._point_due = True
        self._redraw_soon()

    def _on_count(self, k, value):
        self.counts[k] = int(value)
        self.layer = 0
        self._redraw_soon()

    def _on_interpolate(self, on):
        if not on:   # back on the simulated points
            for k, a in enumerate(self.dataset.axes if self.dataset else []):
                values = sorted_values(a["values"])
                self.fixed[k] = float(values[np.argmin(np.abs(values - self.fixed[k]))])
                self.point[k] = float(values[np.argmin(np.abs(values - self.point[k]))])
                self.counts[k] = min(self.counts[k], len(values))
        if self.dataset is not None:
            self._sync_axes()
            self._update_point()
            self.redraw()

    def _redraw_soon(self):
        """Redraw once the slider stops for a moment (dragging would redraw every step)."""
        if not hasattr(self, "_redraw_timer"):
            self._redraw_timer = QTimer(self)
            self._redraw_timer.setSingleShot(True)
            self._redraw_timer.timeout.connect(self._redraw_due)
        self._redraw_timer.start(120)

    def _redraw_due(self):
        if getattr(self, "_point_due", False):
            self._point_due = False
            self._update_point()
        self.redraw()

    # -----------------------------
    # Plots
    # -----------------------------

    def redraw(self):
        self.figure.clear()
        self._picks = {}
        d = self.dataset
        key = self.z_combo.currentData()
        if d is None or key is None:
            self.canvas.draw_idle()
            return
        if not self.plot_axes:
            self.figure.text(0.5, 0.5, "Tick a swept parameter to plot against (Analysis → Plot).", ha="center",
                             va="center", color="#555")
            self.canvas.draw_idle()
            return
        interp = self.interpolate.isChecked()
        fixed = {k: v for k, v in self.fixed.items() if k not in self.plot_axes}
        data = panels(d, key, self.plot_axes, fixed, self.counts, interp)
        self._data = data
        layers = data["layers_data"]
        self.layer = min(max(self.layer, 0), len(layers) - 1)
        grid = layers[self.layer]
        nrows, ncols = len(grid), len(grid[0])
        label, unit = d.labels[key]
        zlabel = f"{label} [{unit}]" if unit else label
        lo, hi = data["zlim"]
        norm = colors.Normalize(lo, hi)
        self._layer_text(data)
        if data["y"] is not None and self.style_combo.currentText() == SURFACE:
            self._draw_surfaces(data, grid, label, zlabel)
            return
        self.plots.setCurrentWidget(self.canvas)
        self.surfaces_view.scroll_layers = False
        axes = []
        for r, row in enumerate(grid):
            for c, panel in enumerate(row):
                ax = self.figure.add_subplot(nrows, ncols, r * ncols + c + 1)
                axes.append(ax)
                self._draw_panel(ax, data, panel, norm, zlabel, r == nrows - 1, c == 0)
                title = self._panel_title(data, panel, r, c)
                if title:
                    ax.set_title(title, fontsize=8)
        if data["y"] is not None:
            self.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes, label=zlabel, shrink=0.8)
        if data["layers"] is not None:
            self.figure.suptitle(self.layer_label.text(), fontsize=9)
        self.canvas.draw_idle()

    def _layer_text(self, data):
        if data["layers"] is not None:
            k, values = data["layers"]
            self.layer_label.setText(f"Layer {self.layer + 1} of {len(values)}: {self._short(k)} = "
                                     f"{fmt(values[self.layer])} (scroll wheel over the plots to change)")

    def _panel_title(self, data, panel, r, c):
        titles = []
        for role in ("columns", "rows"):
            if data[role] is not None and (role == "columns" and r == 0 or role == "rows" and c == 0):
                k = data[role][0]
                titles.append(f"{self._short(k)} = {fmt(panel['at'][k])}")
        return "\n".join(titles)

    def _draw_surfaces(self, data, grid, label, zlabel):
        kx, xs = data["x"]
        ky, ys = data["y"]
        selected = None
        shown = []
        for r, row in enumerate(grid):
            out = []
            for c, panel in enumerate(row):
                out.append(dict(panel, title=self._panel_title(data, panel, r, c)))
                if all(np.isclose(self.point.get(k, v), v) for k, v in panel["at"].items()):
                    selected = (r, c, int(np.argmin(np.abs(xs - self.point[kx]))),
                                int(np.argmin(np.abs(ys - self.point[ky]))))
            shown.append(out)
        if data["layers"] is not None:
            k, values = data["layers"]
            shown[0][0]["title"] = (f"Layer {self.layer + 1}/{len(values)}: {self._short(k)} = "
                                    f"{fmt(values[self.layer])}\n" + shown[0][0]["title"]).strip()
        self.plots.setCurrentWidget(self.surfaces_view)
        self.surfaces_view.scroll_layers = data["layers"] is not None
        self.surfaces_view.draw(shown, xs, ys, data["zlim"], (self._short(kx), self._short(ky), zlabel), selected,
                                label)

    def _on_surface_pick(self, r, c, i, j):
        data = self._data
        panel = data["layers_data"][self.layer][r][c]
        self.point.update(panel["at"])
        self.point[data["x"][0]] = float(data["x"][1][i])
        self.point[data["y"][0]] = float(data["y"][1][j])
        self._update_point()
        self.redraw()

    def _change_layer(self, step):
        self.layer += step
        self.redraw()

    def _short(self, k):
        a = self.dataset.axes[k]
        return f"{a['label']} [{a['unit']}]" if a["unit"] else a["label"]

    def _draw_panel(self, ax, data, panel, norm, zlabel, bottom, left):
        """A curve, or a colour map of z(x, y) (surfaces are drawn by SurfaceGrid)."""
        kx, xs = data["x"]
        z = panel["z"]
        ex, un = panel["extrapolated"], panel["unconverged"]
        fs = 8
        if data["y"] is None:   # a curve
            line, = ax.plot(xs, z, "-o", color="#1565c0", markersize=4, picker=6)
            self._picks[line] = (panel["at"], xs, None)
            ax.plot(xs[ex], z[ex], "o", color="#c62828", markersize=6)
            ax.plot(xs[un], z[un], "x", color="black", markersize=7)
            ax.set_ylim(*_padded(data["zlim"]))
            ax.set_xlabel(self._short(kx), fontsize=fs)
            ax.set_ylabel(zlabel, fontsize=fs)
            ax.grid(True, alpha=0.3)
            self._mark_point(ax, panel, kx, None, xs, None, z)
            return
        ky, ys = data["y"]
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        finite = np.isfinite(z)
        ax.pcolormesh(X, Y, np.ma.masked_invalid(z), cmap=CMAP, norm=norm, shading="nearest")
        pts = ax.scatter(X[finite], Y[finite], s=6, color="k", picker=4)
        ax.scatter(X[ex & finite], Y[ex & finite], s=18, color="#c62828")
        ax.scatter(X[un & finite], Y[un & finite], s=24, marker="x", color="black")
        ax.tick_params(labelsize=7)
        if bottom:
            ax.set_xlabel(self._short(kx), fontsize=fs)
        if left:
            ax.set_ylabel(self._short(ky), fontsize=fs)
        self._picks[pts] = (panel["at"], X[finite], Y[finite])
        self._mark_point(ax, panel, kx, ky, xs, ys, z)

    def _mark_point(self, ax, panel, kx, ky, xs, ys, z):
        """The point shown in 3D, if it lies in this panel."""
        if any(not np.isclose(self.point.get(k, v), v) for k, v in panel["at"].items()):
            return
        px = self.point.get(kx)
        if ky is None:
            if px is None:
                return
            ax.plot([px], [np.interp(px, xs, z)], "o", markersize=12, markerfacecolor="none",
                    markeredgecolor="#ff6f00", markeredgewidth=2)
            return
        ax.scatter([px], [self.point.get(ky)], s=90, facecolors="none", edgecolors="#ff6f00", linewidths=2)

    def _on_plot_pick(self, event):
        info = self._picks.get(event.artist)
        if info is None or not len(event.ind):
            return
        at, xs, ys = info
        i = int(event.ind[0])
        self.point.update(at)
        if ys is None:
            self.point[self.plot_axes[0]] = float(xs[i])
        else:
            self.point[self.plot_axes[0]] = float(np.ravel(xs)[i])
            self.point[self.plot_axes[1]] = float(np.ravel(ys)[i])
        self._update_point()
        self.redraw()

    def _on_scroll(self, event):
        if len(self.plot_axes) < len(ROLES):
            return
        self._change_layer(1 if event.button == "up" else -1)

    # -----------------------------
    # The point in 3D
    # -----------------------------

    def _update_point(self):
        d = self.dataset
        if d is None:
            return
        for k in range(len(d.axes)):
            if k not in self.plot_axes:
                self.point[k] = self.fixed[k]
        interp = self.interpolate.isChecked()
        values, coords, extrapolated, converged = evaluate_point(d, self.point, interp, self.shapes)
        params = [self.point[k] for k in range(len(d.axes))]
        warnings = ["The pre-activation Δp is outside the design's simulated range: EXTRAPOLATING, not "
                    "interpolating."] if extrapolated else []
        values["warnings"] = warnings
        self.rows = [{"index": None, "params": params, "converged": converged, "time": np.nan, "values": values,
                      "message": "", "coords": coords}]
        self.row = 0
        lines = [f"{self._short(k)} = {fmt(v)}" for k, v in enumerate(params)]
        key = self.z_combo.currentData()
        if key is not None:
            label, unit = d.labels[key]
            lines.append(f"<b>{label} = {fmt(values[key])} {unit}</b>")
        if not converged:
            lines.append("<span style='color:#c62828'>not converged (or not solved) here</span>")
        if interp:
            lines.append("<i>interpolated</i>")
        self.point_label.setText("<br>".join(lines))
        if self.selection is not None and self.selection[0] != "group":
            self._show_properties()
        self.refresh_view()

def _padded(lim):
    lo, hi = lim
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad
