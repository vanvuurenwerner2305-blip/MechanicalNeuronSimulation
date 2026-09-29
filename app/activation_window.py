"""
Activation-function space: import the valve CAD, assign roles, set the flow connections, simulate the
Δp sweep (structure and gas flow together) and save the result as an activation-function design
(*.mad) that can be used as a part without simulating it again.
"""
import csv

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                            QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                            QMessageBox, QPushButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                            QWidget)

from membrane_sim.flow import ORIFICE_LAW, SEGMENT_LAW, compile_flow_law

from .activation import (STUDY_FIELDS, ActivationProject, build_activation, detect_connections,
                         generate_activation_mesh, run_study)
from .main_window import MESH, MODEL, RESULTS, MainWindow
from .panels import SolverPanel, fmt
from .project import (ACTIVATION_ROLES, CLOSED, CONNECTION_TYPES, DYNAMIC_FLUID, FLOW_LAW_HELP, FLUID, OPENING,
                      ORIFICE)
from .viewport import RESULT_FIELDS, polydata

TITLE = "Activation Function Simulator"
CONNECTION_COLORS = {OPENING: "#2ca02c", ORIFICE: "#d35400", CLOSED: "#999999"}

GUIDE = """
<h3>Activation function workflow</h3>
<p>The device maps the pressure difference <b>Δp</b> across a membrane to how far a soft tube is
squeezed shut, and so to the gas flow through it and the pressures along it. The membrane pushes,
through a part bonded to it (e.g. a pusher), on the tube and squeezes it against a rigid body.
Simulate it once, save it as a design (<b>*.mad</b>) and use the mapping as a part.</p>
<ol>
<li><b>Model in CAD</b> with every region as a solid body: the membrane (thin solid), the part
between membrane and tube (e.g. a pusher touching the membrane's face), the tube, the rigid body
behind it, and the gas: one body filling the inside of the tube, a supply body against its inlet
end and a sink body (e.g. 0 kPa) against its outlet end. Export STEP in mm.</li>
<li><b>File → Open STEP</b> and assign roles (Edit → Auto-assign guesses them from names):
<ul>
<li><b>Membrane</b> / <b>Shell</b>: clamped along its edges. Δp pushes it towards the tube. It is bonded to
every free rigid body or solid its face touches.</li>
<li><b>Channel (tube)</b>: simulated as a 3D solid, fixed at its two end faces.</li>
<li><b>Fluid</b>, <i>Constant pressure</i>: a supply or sink at a set pressure.
<i>Dynamic pressure</i>: its pressure follows from the flow. The one inside the tube is cut into
<b>Segments</b>, each a flow resistance with its own pressure on the tube wall.</li>
<li><b>Rigid body</b>: <i>Fixed</i> or <i>Free</i> (moves and tilts, e.g. a rigid pusher).
<b>Solid</b>: a deformable part, e.g. a soft pusher.</li>
</ul></li>
<li><b>Flow connections</b> appear in the model tree wherever two fluids touch (and where a dynamic fluid
faces the outside). Click one to make it an <b>Opening</b> (no resistance), an <b>Orifice</b> (your
equation) or <b>Closed</b>.</li>
<li>Set the Δp range and the gas in the <b>Study</b> tab and press <b>Simulate</b> (F5). At every Δp the
structure and the flow are iterated until the wall pressures stop changing.</li>
<li><b>File → Save design</b> keeps everything and the computed mapping in one .mad file.</li>
</ol>
<p>Resistance equations give the pressure drop in Pa for a mass flow mdot (kg/s), in SI units, with
the gas density rho from the ideal gas law. Units elsewhere: mm, N, MPa; pressures in kPa.</p>
"""


def _travel(results):
    return results.get("travel", [float("nan")] * len(results["dp"]))


# -----------------------------
# Flow connections panel
# -----------------------------

class FlowPanel(QWidget):
    """The flow connections found between fluid bodies, with their type and resistance equation, and the
    resistance equation of the tube's segments."""
    changed = Signal(str, str, object)   # connection key, "type" | "law", value
    selected = Signal(str)               # connection key ("" = none)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        intro = QLabel("Where two fluid bodies touch, gas flows between them. Choose how: an opening (no "
                       "resistance), an orifice (your equation) or closed. The fluid inside the tube is split "
                       "into segments; their resistance is a property of that fluid (Part tab).")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555;")
        layout.addWidget(intro)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._row_changed)
        layout.addWidget(self.list, 1)

        box = QGroupBox("Connection")
        form = QFormLayout(box)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        form.addRow(self.info)
        self.type = QComboBox()
        self.type.addItems(list(CONNECTION_TYPES))
        self.type.activated.connect(lambda _: self._emit("type", self.type.currentText()))
        form.addRow("Type", self.type)
        self.law = QLineEdit()
        self.law.setToolTip(FLOW_LAW_HELP)
        self.law.editingFinished.connect(lambda: self._emit("law", self.law.text()))
        form.addRow("Δp [Pa] =", self.law)
        self.law_error = QLabel("")
        self.law_error.setStyleSheet("color: #c0392b;")
        self.law_error.setWordWrap(True)
        form.addRow(self.law_error)
        help_ = QLabel(FLOW_LAW_HELP.replace("\n", "<br>"))
        help_.setWordWrap(True)
        help_.setStyleSheet("color: #555; font-size: 11px;")
        form.addRow(help_)
        layout.addWidget(box)
        self.box = box
        self.connections, self.settings = [], {}
        self.set_connections([], {})

    def set_connections(self, connections, settings):
        key = self.current_key()
        self.connections, self.settings = connections, settings
        self.list.blockSignals(True)
        self.list.clear()
        for c in connections:
            s = settings.get(c.key, {})
            QListWidgetItem(f"{c.key}   ({s.get('type', OPENING)})", self.list)
        self.list.blockSignals(False)
        keys = [c.key for c in connections]
        self.list.setCurrentRow(keys.index(key) if key in keys else (0 if keys else -1))
        self._row_changed(self.list.currentRow())

    def current_key(self):
        row = self.list.currentRow()
        return self.connections[row].key if 0 <= row < len(self.connections) else ""

    def select(self, key):
        keys = [c.key for c in self.connections]
        if key in keys:
            self.list.setCurrentRow(keys.index(key))

    def _row_changed(self, row):
        has = 0 <= row < len(self.connections)
        self.box.setEnabled(has)
        if not has:
            self.info.setText("No flow connections: assign Fluid roles to the gas bodies.")
            self.selected.emit("")
            return
        c = self.connections[row]
        s = self.settings.get(c.key, {})
        self.info.setText(f"<b>{c.key}</b><br>Contact area A = {c.area:.4g} mm², perimeter P = {c.perimeter:.4g} mm, "
                          f"{c.width:.3g} × {c.height:.3g} mm")
        self.type.setCurrentText(s.get("type", OPENING))
        self.law.setText(s.get("law", ORIFICE_LAW))
        self.law.setEnabled(s.get("type") == ORIFICE)
        self._check_law()
        self.selected.emit(c.key)

    def _check_law(self):
        try:
            compile_flow_law(self.law.text(), ORIFICE_LAW)
            self.law_error.setText("")
        except Exception as exc:  # shown next to the field
            self.law_error.setText(str(exc))

    def _emit(self, what, value):
        key = self.current_key()
        if key:
            self.changed.emit(key, what, value)
            self.law.setEnabled(self.type.currentText() == ORIFICE)
            item = self.list.currentItem()
            if item is not None:
                item.setText(f"{key}   ({self.type.currentText()})")
            self._check_law()


# -----------------------------
# Results
# -----------------------------

class ActivationResultsPanel(QWidget):
    """A(Δp) and mass flow(Δp), the area and pressure along the tube at the selected point, and display
    settings for the 3D view (same signals and current_step() as ResultsPanel)."""
    display_changed = Signal()
    export_vtk = Signal()
    export_csv = Signal()
    screenshot = Signal()
    save_design = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.status = QLabel("No results yet. Press Simulate (F5).")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.figure = Figure(figsize=(4, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(540)
        self.ax_map = self.figure.add_subplot(3, 1, 1)
        self.ax_ends = self.figure.add_subplot(3, 1, 2, sharex=self.ax_map)
        self.ax_profile = self.figure.add_subplot(3, 1, 3)
        self.ax_flow = self.ax_map.twinx()
        self.ax_pressure = self.ax_profile.twinx()
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Δp [kPa]", "A [mm²]", "ṁ [g/s]", "p end [kPa]", "Travel [mm]", ""])
        self.table.horizontalHeaderItem(3).setToolTip("Pressure at the downstream end of the tube")
        self.table.horizontalHeaderItem(4).setToolTip("How far the part bonded to the membrane moved towards the tube")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._table_selected)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.canvas)
        split.addWidget(self.table)
        split.setSizes([600, 150])
        layout.addWidget(split, 1)

        display = QGroupBox("3D view")
        form = QFormLayout(display)
        self.field = QComboBox()
        self.field.addItems(RESULT_FIELDS)
        self.field.currentTextChanged.connect(lambda _: self.display_changed.emit())
        form.addRow("Field", self.field)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.0, 1000.0)
        self.scale.setDecimals(2)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.25)
        self.scale.valueChanged.connect(lambda _: self.display_changed.emit())
        form.addRow("Deformation scale", self.scale)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.step = QSlider(Qt.Horizontal)
        self.step.valueChanged.connect(self._step_changed)
        self.step_label = QLabel("")
        self.step_label.setMinimumWidth(80)
        h.addWidget(self.step, 1)
        h.addWidget(self.step_label)
        form.addRow("Δp point", row)
        self.fixed_range = QCheckBox("Same colour range for all points")
        self.fixed_range.setChecked(True)
        self.fixed_range.toggled.connect(lambda _: self.display_changed.emit())
        form.addRow(self.fixed_range)
        layout.addWidget(display)

        buttons = QHBoxLayout()
        for text, signal in (("Save design…", self.save_design), ("Export CSV…", self.export_csv),
                             ("Export VTK…", self.export_vtk), ("Screenshot…", self.screenshot)):
            b = QPushButton(text)
            b.clicked.connect(signal.emit)
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.results = None   # dict as stored in the design
        self.states = []      # per point: 3D state (only after simulating in this session)
        self._updating = False
        self.set_results(None, [], "No results yet. Press Simulate (F5).")

    # -----------------------------

    def set_results(self, results, states=(), status_text=""):
        if isinstance(results, list):  # MainWindow's reset call: set_results([], text)
            results, states, status_text = None, [], states if isinstance(states, str) else status_text
        self.results, self.states = results, list(states)
        self.status.setText(status_text)
        n = len(results["dp"]) if results else 0
        self._updating = True
        self.step.setRange(0, max(0, n - 1))
        self.step.setValue(n - 1 if n else 0)
        self.table.setRowCount(n)
        for r in range(n):
            values = [results["dp"][r], results["area"][r], 1000.0 * results["mdot"][r], results["p_end"][r],
                      _travel(results)[r]]
            for c, v in enumerate(values):
                item = QTableWidgetItem(fmt(v))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
            self.table.setItem(r, 5, QTableWidgetItem("" if results["converged"][r] else "not converged"))
        self._updating = False
        self._update_label()
        self.plot()

    def set_table(self, rows, text=""):
        pass  # the table lists the sweep points (set_results)

    def add_point(self, results_so_far):
        """Live update while the sweep runs: follow the newest point (the slider may still hold a point of the
        previous, longer run)."""
        self.results, self.states = results_so_far, []
        n = len(results_so_far["dp"])
        self._updating = True
        self.step.setRange(0, max(0, n - 1))
        self.step.setValue(n - 1)
        self._updating = False
        self._update_label()
        self.plot()

    def current_index(self):
        if not self.results or not len(self.results["dp"]):
            return None
        return min(max(self.step.value(), 0), len(self.results["dp"]) - 1)

    def current_step(self):
        k = self.current_index()
        return self.states[k] if self.states and k is not None and k < len(self.states) else None

    def _step_changed(self, _):
        if self._updating:
            return
        self._update_label()
        self._updating = True
        self.table.selectRow(self.step.value())
        self._updating = False
        self.plot()
        self.display_changed.emit()

    def _table_selected(self):
        if self._updating:
            return
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.step.setValue(rows[0].row())

    def _on_click(self, event):
        if event.inaxes not in (self.ax_map, self.ax_flow, self.ax_ends) or not self.results or event.xdata is None:
            return
        dp = np.asarray(self.results["dp"])
        self.step.setValue(int(np.argmin(np.abs(dp - event.xdata))))

    def _update_label(self):
        k = self.current_index()
        self.step_label.setText(f"{self.results['dp'][k]:.4g} kPa" if k is not None else "")

    def plot(self):
        for ax in (self.ax_map, self.ax_flow, self.ax_ends, self.ax_profile, self.ax_pressure):
            ax.clear()
        r = self.results
        if not r or not r.get("dp"):
            self.ax_map.set_title("No results yet", fontsize=9)
            self.canvas.draw_idle()
            return
        dp, A, mdot = np.asarray(r["dp"]), np.asarray(r["area"]), 1000.0 * np.asarray(r["mdot"])
        ok = np.asarray(r["converged"], bool)
        purple, blue, green = "#6a3d9a", "#1f77b4", "#2ca02c"
        self.ax_map.plot(dp, A, "-o", color=purple, ms=4)
        if (~ok).any():
            self.ax_map.plot(dp[~ok], A[~ok], "x", color="#d62728", ms=8)
        if "A0" in r:
            self.ax_map.axhline(r["A0"], color=purple, lw=0.8, ls=":")
        self.ax_map.set_xlabel("Δp across the membrane [kPa]", fontsize=8)
        self.ax_map.set_ylabel("A smallest section [mm²]", color=purple, fontsize=8)
        self.ax_map.set_ylim(bottom=0)
        self.ax_flow.plot(dp, mdot, "-s", color=blue, ms=3, lw=1)
        self.ax_flow.set_ylabel("ṁ through the tube [g/s]", color=blue, fontsize=8)
        self.ax_flow.yaxis.set_label_position("right")   # clear() puts the twin's axis back on the left
        self.ax_flow.yaxis.tick_right()
        k = self.current_index()
        if k is not None:
            self.ax_map.axvline(dp[k], color="#999", lw=0.8)
            self.ax_map.plot([dp[k]], [A[k]], "o", ms=9, mfc="none", mec="#d35400", mew=2)
        for ax in (self.ax_map, self.ax_flow):
            ax.tick_params(labelsize=7)

        # pressures at the two ends of the dynamic fluid in the tube, against the membrane pressure
        pressures = r.get("pressures", [])
        if pressures and len(pressures) == len(dp):
            P = np.asarray([row for row in pressures], float)
            self.ax_ends.plot(dp, P[:, 0], "-o", color="#2ca02c", ms=4, label="upstream end")
            self.ax_ends.plot(dp, P[:, -1], "-s", color="#d62728", ms=4, label="downstream end")
            if (~ok).any():
                self.ax_ends.plot(dp[~ok], P[~ok, -1], "x", color="#d62728", ms=8)
            if k is not None:
                self.ax_ends.axvline(dp[k], color="#999", lw=0.8)
            self.ax_ends.set_ylabel("Gas pressure at the\ntube ends [kPa]", fontsize=8)
            self.ax_ends.set_xlabel("Δp across the membrane [kPa]", fontsize=8)
            self.ax_ends.legend(fontsize=7, loc="best")
            self.ax_ends.grid(True, lw=0.3, alpha=0.5)
            self.ax_ends.tick_params(labelsize=7)

        s = np.asarray(r.get("stations", []))
        if len(s) and k is not None:
            self.ax_profile.plot(s, r["rest_profile"], color="#aaa", lw=1)
            if k < len(r.get("profiles", [])):
                self.ax_profile.plot(s, r["profiles"][k], color=purple, lw=1.5)
            self.ax_profile.set_ylabel("Section area [mm²]", color=purple, fontsize=8)
            self.ax_profile.set_ylim(bottom=0)
            if k < len(r.get("pressures", [])) and "node_positions" in r:
                self.ax_pressure.plot(r["node_positions"], r["pressures"][k], "-o", color=green, ms=3, lw=1.2)
                self.ax_pressure.set_ylabel("Gas pressure [kPa]", color=green, fontsize=8)
                self.ax_pressure.yaxis.set_label_position("right")
                self.ax_pressure.yaxis.tick_right()
            self.ax_profile.set_xlabel(f"Along the tube, upstream → downstream [mm]   (Δp = {dp[k]:.4g} kPa)",
                                       fontsize=8)
            for ax in (self.ax_profile, self.ax_pressure):
                ax.tick_params(labelsize=7)
        self.canvas.draw_idle()


# -----------------------------
# Window
# -----------------------------

class ActivationWindow(MainWindow):
    SPACE_ROLES = ACTIVATION_ROLES
    PROJECT_CLASS = ActivationProject
    PROJECT_FILTER = "Activation function design (*.mad)"
    TITLE = TITLE

    def __init__(self):
        self.states = []
        self.connections = []      # detected flow connections (activation.Connection)
        self.highlight = None      # (body index, faces) of the selected connection
        super().__init__()
        self.log.clear()
        self.log_message("Activation function space: open the valve's STEP file (File → Open STEP…). "
                         "Help → Quick guide explains the roles and the workflow.")

    # -----------------------------
    # UI
    # -----------------------------

    def _make_results_panel(self):
        panel = ActivationResultsPanel()
        panel.save_design.connect(lambda: self.save_project(False))
        return panel

    def _extra_tabs(self):
        self.study_panel = SolverPanel(title="Δp sweep and gas", fields=STUDY_FIELDS, note=(
            "Δp is the pressure difference across the membrane; positive Δp pushes it towards the tube. "
            "The gas density follows the ideal gas law, ρ = p_abs / (R T), at the local pressure."))
        self.study_panel.changed.connect(self._on_study_changed)
        self.flow_panel = FlowPanel()
        self.flow_panel.changed.connect(self._on_connection_changed)
        self.flow_panel.selected.connect(self._on_connection_selected)
        return [(self.flow_panel, "Flow"), (self.study_panel, "Study")]

    def _build_actions(self):
        super()._build_actions()
        self.a_solve.setText("Simulate")
        self.a_solve.setStatusTip("Run the Δp sweep and compute the activation function")
        self.a_solve.setToolTip("Run the Δp sweep and compute the activation function")
        self.a_save.setText("Save design")
        self.a_save_as.setText("Save design as…")
        self.a_open_project.setText("Open design…")
        self.a_sweep.setVisible(False)

    def _on_study_changed(self, key, value):
        setattr(self.project.study, key, value)
        self._invalidate(mesh=False)

    def _after_load(self):
        self.study_panel.load(self.project.study)
        self.states = []
        self._refresh_connections()
        results = self.project.results
        if results:
            n, ok = len(results["dp"]), sum(results["converged"])
            self.results_panel.set_results(results, [], f"Saved activation function: {ok}/{n} points "
                                                        f"(Simulate again to see the deformed device).")
            self.tabs.setCurrentWidget(self.results_panel)
            self.log_message(f"Loaded design with {n} simulated points (Δp {results['dp'][0]:.4g} … "
                             f"{results['dp'][-1]:.4g} kPa); no simulation needed to use it.")
        else:
            self.results_panel.set_results(None, [], "No results yet. Press Simulate (F5).")

    def _couplings_text(self, indices):
        return None

    def show_guide(self):
        box = QMessageBox(self)
        box.setWindowTitle("Quick guide - activation function")
        box.setTextFormat(Qt.RichText)
        box.setText(GUIDE)
        box.exec()

    # -----------------------------
    # Flow connections
    # -----------------------------

    def _refresh_connections(self):
        """Find where fluids touch (on the display meshes) and list the connections in the tree and Flow tab."""
        if not self.surfaces:
            return
        try:
            self.connections = detect_connections(self.surfaces, self.project.parts, self.cad.bodies)
        except Exception as exc:  # geometry problems should not block editing
            self.connections = []
            self.log_message(f"Could not find the flow connections: {exc}")
        for c in self.connections:
            self.project.connection(c.key, outside=c.b is None)
        settings = self.project.connections
        self.tree.set_connections([(c.key, settings[c.key]["type"].split(" (")[0],
                                    CONNECTION_COLORS[settings[c.key]["type"]]) for c in self.connections])
        self.flow_panel.set_connections(self.connections, settings)

    def _on_role_changed(self, indices, role):
        super()._on_role_changed(indices, role)
        QTimer.singleShot(0, self._refresh_connections)

    def _on_props_changed(self, indices, key, value):
        super()._on_props_changed(indices, key, value)
        if key == "model":  # constant/dynamic changes which outside connections exist
            QTimer.singleShot(0, self._refresh_connections)

    def auto_assign(self):
        super().auto_assign()
        self._refresh_connections()

    def _on_connection_changed(self, key, what, value):
        self.project.connections.setdefault(key, {"type": OPENING, "law": ORIFICE_LAW})[what] = value
        self._invalidate(mesh=False)
        if what == "type":
            settings = self.project.connections
            self.tree.set_connections([(c.key, settings[c.key]["type"].split(" (")[0],
                                        CONNECTION_COLORS[settings[c.key]["type"]]) for c in self.connections])

    def _on_connection_selected(self, key):
        c = next((c for c in self.connections if c.key == key), None)
        self.highlight = (c.a, c.faces) if c is not None else None
        if self.mode != RESULTS:
            self.refresh_view()

    def set_selection(self, indices, from_tree=False):
        conns = [i for i in indices if i < 0]
        if conns:  # a flow connection in the tree
            k = -1 - conns[0]
            if k < len(self.connections):
                self.tabs.setCurrentWidget(self.flow_panel)
                self.flow_panel.select(self.connections[k].key)
            indices = [i for i in indices if i >= 0]
        super().set_selection(indices, from_tree)
        if conns:
            self.tabs.setCurrentWidget(self.flow_panel)

    # -----------------------------
    # Jobs
    # -----------------------------

    def generate_mesh(self):
        def job(worker, cad, project):
            worker.report(-1, "Meshing…")
            data = generate_activation_mesh(cad, project)
            return data, build_activation(cad, data, project)
        self._start(job, (self.cad, self.project), self._mesh_done, "Generating mesh…")

    def _mesh_done(self, result):
        data, check = result
        self.adopt_mesh(data)
        self.check = check
        self._report(check)
        self.mode = MESH
        self.view_actions[MESH].setChecked(True)
        self.set_selection(self.selection)

    def check_model(self):
        self.generate_mesh()

    def _report(self, b):
        parts = self.project.parts
        tube, f = b.tube, b.flow
        self.log_message("Model:")
        self.log_message(f"  Tube: {len(tube.tets_np)} {'10' if tube.order == 2 else '4'}-node tetrahedra, "
                         f"{tube.n_nodes} nodes, {b.fixed_tube_nodes} fixed (end faces)")
        self.log_message(f"  Gas in the tube: {parts[f.lumen].name}, {f.segments} segments over "
                         f"{f.bounds[-1] - f.bounds[0]:.4g} mm; inside height {b.channel_height:.4g} mm")
        self.log_message(f"  Cross-section at rest A0 = {b.A0:.5g} mm² (smallest of {len(b.sections.stations)} sections)")
        for conn, st in f.connections:
            self.log_message(f"  {conn.key}: {st['type']}" + (f", Δp = {st['law']}" if st["type"] == ORIFICE else ""))
        for i, t in b.thickness.items():
            self.log_message(f"  {parts[i].name}: thickness {t:.4g} mm")
        for i, j, n in b.bonds:
            self.log_message(f"  {parts[i].name} bonded to {parts[j].name} ({n} nodes)")
        for j, body in b.movers.items():
            kind = "free rigid body" if getattr(body, "is_rigid", False) else \
                f"solid, {len(body.tets_np)} tetrahedra, {int(body.fixed.sum())} fixed nodes"
            self.log_message(f"  {parts[j].name}: {kind}")
        for w in b.warnings:
            self.log_message("  Warning: " + w)

    def _check_laws(self):
        """Compile every resistance equation before a long run; returns an error message or ''."""
        for part in self.project.parts:
            if part.role == FLUID and part.props.get("model") == DYNAMIC_FLUID:
                try:
                    compile_flow_law(part.props.get("segment_law"), SEGMENT_LAW)
                except Exception as exc:
                    return f"{part.name}, segment resistance: {exc}"
        for key, st in self.project.connections.items():
            if st.get("type") == ORIFICE:
                try:
                    compile_flow_law(st.get("law"), ORIFICE_LAW)
                except Exception as exc:
                    return f"{key}: {exc}"
        return ""

    def solve(self):
        error = self._check_laws()
        if error:
            QMessageBox.warning(self, TITLE, f"A flow resistance equation is not valid:\n{error}")
            return

        def job(worker, cad, project, mesh):
            if mesh is None:
                worker.report(-1, "Meshing…")
                mesh = generate_activation_mesh(cad, project)
            build = build_activation(cad, mesh, project)
            for w in build.warnings:
                worker.log("Warning: " + w)
            rows = []

            def callback(k, n, dp, lam, iteration, residual):
                worker.check()
                worker.report((k + lam) / n, f"Δp = {dp:.4g} kPa ({k + 1}/{n}): load {lam:.0%}, "
                                             f"Newton iteration {iteration}, |R| = {residual:.2e}")

            def point(row):
                rows.append(row)
                worker.log(f"  Δp = {row['dp']:.4g} kPa: A = {row['area']:.5g} mm², ṁ = {1000 * row['mdot']:.4g} g/s, "
                           f"p end = {row['p_end']:.4g} kPa, travel {row['travel']:.4g} mm, "
                           f"{row['iterations']} flow iterations" + ("" if row["converged"] else "  (NOT converged)"))
                worker.item.emit((build, list(rows)))

            results, states = run_study(build, project, callback=callback, point_callback=point, check=worker.check)
            return mesh, build, results, states

        self._start(job, (self.cad, self.project, self.mesh_data_if_current()), self._solve_done,
                    "Simulating the activation function…")
        self.worker.item.connect(self._on_point)
        self.tabs.setCurrentWidget(self.results_panel)

    def _on_point(self, item):
        build, rows = item
        partial = {k: [r[k] for r in rows] for k in ("dp", "area", "mdot", "p_end", "travel", "converged")}
        partial.update(profiles=[r["profile"] for r in rows], pressures=[r["pressures"] for r in rows],
                       stations=(build.sections.stations - build.flow.bounds[0]).tolist(),
                       node_positions=build.flow.node_positions().tolist(),
                       rest_profile=build.sections.rest_areas.tolist(), A0=build.A0)
        self.results_panel.add_point(partial)

    def _solve_done(self, result):
        data, build, results, states = result
        self.adopt_mesh(data)
        self.build = self.check = build
        self.states = states
        self.project.results = results
        self.results_outdated = False
        self._clim = None
        n, ok = len(results["dp"]), sum(results["converged"])
        status = f"Activation function: {ok}/{n} points converged · A0 = {results['A0']:.4g} mm² · " \
                 f"at Δp = {results['dp'][-1]:.4g} kPa: A = {results['area'][-1]:.4g} mm², " \
                 f"ṁ = {1000 * results['mdot'][-1]:.4g} g/s"
        self.log_message(status)
        if ok < n:
            QMessageBox.warning(self, TITLE, f"{n - ok} of {n} points did not converge; they are marked in the plot.")
        self.results_panel.set_results(results, states, status + "\nSave the design (Ctrl+S) to keep it.")
        self.tabs.setCurrentWidget(self.results_panel)
        self.set_mode(RESULTS)

    # -----------------------------
    # Views
    # -----------------------------

    def refresh_view(self):
        if self.mode == RESULTS and (self.build is None or self.results_panel.current_step() is None):
            self.mode = MODEL
            self.view_actions[MODEL].setChecked(True)
        super().refresh_view()
        if self.highlight is not None and self.mode != RESULTS and self.highlight[0] in self.surfaces:
            index, faces = self.highlight
            m = self.surfaces[index]
            if len(faces) and self.mode == MODEL:
                self.viewport._add("connection", polydata(m.vertices, m.faces[faces]), color="#ffcc00", opacity=1.0,
                                   show_edges=False, pickable=False)
                self.viewport.plotter.render()

    def _global_clim(self, field):
        key = (id(self.build), field)
        if self._clim is None or self._clim[0] != key:
            values = [v for step in self.states for v, _ in self.viewport.result_values(self.build, step, field).values()]
            allv = np.concatenate(values)
            lo, hi = float(allv.min()), float(allv.max())
            self._clim = (key, (lo, hi if hi > lo else lo + 1e-12))
        return self._clim[1]

    def _fill_results_table(self, step):
        pass

    def set_mode(self, mode):
        if mode == RESULTS and (self.build is None or not self.states):
            mode = MODEL
        super().set_mode(mode)

    def _update_actions(self):
        super()._update_actions()
        self.view_actions[RESULTS].setEnabled(self.build is not None and bool(self.states))

    # -----------------------------
    # Files
    # -----------------------------

    def save_project(self, save_as=False):
        if self.project.results and self.results_outdated:
            answer = QMessageBox.question(
                self, TITLE, "The model changed after the last simulation. Save the design with the old results?\n\n"
                             "(No: save without results; simulate again to compute them.)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.No:
                self.project.results = None
        super().save_project(save_as)

    def export_csv(self):
        r = self.project.results
        if not r:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export activation function", "activation_function.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            positions = r.get("node_positions", [])
            writer.writerow(["dp [kPa]", "A [mm2]", "mdot [kg/s]", "p end [kPa]", "travel [mm]", "converged"]
                            + [f"p at {x:.3g} mm [kPa]" for x in positions])
            for k in range(len(r["dp"])):
                writer.writerow([r["dp"][k], r["area"][k], r["mdot"][k], r["p_end"][k], _travel(r)[k],
                                 r["converged"][k]] + list(r["pressures"][k]))
        self.log_message(f"Exported {path}")

    def export_vtk(self):
        step = self.results_panel.current_step()
        if self.build is None or step is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export deformed device", "activation.vtm", "VTK multiblock (*.vtm)")
        if not path:
            return
        blocks = pv.MultiBlock()
        for (i, body), coords in zip(self.build.shells.items(), step["shell_coords"]):
            pd = polydata(coords.numpy(), body.faces.cpu().numpy())
            pd.point_data["displacement"] = coords.numpy() - body.X.cpu().numpy()
            blocks[self.project.parts[i].name] = pd
        blocks.save(path)
        self.log_message(f"Exported {path}")
