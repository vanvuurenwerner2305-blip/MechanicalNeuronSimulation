"""
Full-neuron workbench window: import an inputs-to-pre-activation project and a pre-activation-to-activation
design, link them, change fluid parameters, characterise the neuron over any number of swept parameters (the
dataset is kept in the .mfn), and look at both models side by side (each with its own position and rotation, set by
clicking its group in the model tree).
"""
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
from qtpy.QtCore import QSettings, Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                            QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
                            QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                            QMessageBox, QPlainTextEdit, QProgressBar,
                            QPushButton, QScrollArea, QSlider, QSpinBox, QStyle, QTableWidget,
                            QTableWidgetItem, QTabWidget, QToolBar, QToolButton, QTreeWidget, QTreeWidgetItem,
                            QVBoxLayout, QWidget)

try:
    from qtpy.QtWidgets import QAction
except ImportError:  # Qt6
    from qtpy.QtGui import QAction

from .activation import channel_axis, detect_connections, display_bodies, generate_activation_mesh, segment_faces
from .activation_window import OUTPUT_COLORS, _raised
from .builder import generate_mesh, mesh_sizes
from .cad import CadModel
from .full_neuron import (ACTIVATION, CURRENT, MISSING, MOVE_FIELDS, NEURON, OUTDATED, PREACTIVATION, SHEET_FIELDS,
                          SIZE_FIELDS, SOLID_FIELDS, SOLVER_KEY, SUFFIX, target, FullNeuronProject, parts_of, axis_parts, default_axis, editable_fields, make_dataset,
                          run_grid, run_valve_grid, shell_bodies, transform_points, valve_fields, values_or_nan)
from .panels import FieldForm, SolverPanel, color_icon, fmt
from .project import (ACTIVATION_MEMBRANE, AUTOMATIC, CHAMBER, DEFORMABLE, FLUID, NEURON_SOLVER_FIELDS, RIGID,
                      ROLE_COLORS, ROLE_FIELDS, VALVE_SOLVER_FIELDS, Field, PartSettings, part_color)
from .viewport import Viewport, polydata, solid_shell
from .workers import Worker

TITLE = "Full neuron"
FILTER = f"Full neuron (*{SUFFIX})"
DESIGN_BASE = 100000     # pick ids of the design's parts: DESIGN_BASE + index
GROUP_NAMES = {NEURON: "Inputs to pre-activation", ACTIVATION: "Pre-activation to activation"}
LINK_COLOR = "#c2185b"
STATUS_COLORS = {OUTDATED: "#e65100", MISSING: "#c62828"}
WATCH_MS = 1500          # how often the source files are checked for saved changes
MODEL, RESULTS = "Model", "Results"
# valve on its own
DP_FIELD = Field("dp", "Pre-activation Δp (when not swept)", "float", 0.0, "kPa", -1e6, 1e6, 4,
                 tooltip="Pressure difference across the valve's membrane (positive pushes it towards the tube).")
AREA_FIELD = Field("area", "Orifice area", "float", 0.0, "mm²", 0.0, 1e6, 5,
                   tooltip="The area A its law sees. 0: the contact face between the two fluids (often the whole bore).")

GUIDE = """
<h3>Full neuron</h3>
<p>Put the two halves of a neuron together and solve it as a whole.</p>
<ol>
<li><b>File → Import inputs to pre-activation</b>: a neuron project (*.mns) from the first tab.
<b>File → Import pre-activation to activation</b>: a simulated design (*.mad) from the second tab.</li>
<li><b>Link</b> tab: choose the neuron membrane that is the design's membrane, and the chamber that drives it.
That membrane is not simulated: the design's pre-simulated response replaces it.</li>
<li>Click a fluid in the model tree to change its parameters (pressure, volumes, stiffness, ...). The kinds of
parts and chambers, and the design, are fixed here (change those in their own tabs).</li>
<li>Click a group (<i>Inputs to pre-activation</i> / <i>Pre-activation to activation</i>) to set its position and
rotation in the view.</li>
<li><b>Record &amp; run</b> tab: <b>Add parameter</b> for everything to sweep (any chamber's pressure, ghost
volume, liquid share, stiffness or fluid volume; from, to, points) and tick everything to record. <b>Run
characterisation</b> (F6) solves every combination and keeps the dataset with the model (saved in the *.mfn), so the
neuron can later be used without simulating it again. <b>Solve</b> (F5) solves once at the set values. Click a row in
the Results tab to see that point in 3D (neuron and design coloured by displacement).</li>
<li><b>Valve on its own</b>: import only a pre-activation-to-activation design (no neuron). Then the valve
is simulated again at every point, so <b>Add parameter</b> can sweep the Δp across its membrane
(<i>Pre-activation Δp</i>), its supply and sink pressures, its orifice areas (orifices are listed under the design in
the model tree) and the materials of its membrane, tube and solid parts.</li>
<li><b>File → Save</b> keeps everything in one *.mfn file (copies of both components are stored in it; the original
.mns and .mad are not changed).</li>
</ol>
<p><b>Out of date components</b> (as in a CAD assembly): the full neuron keeps using the versions of the .mns and
.mad it imported. When you save changes to one of them (in its own tab or elsewhere), its group in the model tree is
marked <span style="color:#e65100">out of date</span> and <b>Update</b> appears in the toolbar. Updating takes the new
version; fluid parameters and solver settings you changed here are kept (you are asked about values changed in both
places). Right-click a group to update it or to open its file in its own tab.</p>
"""


class TransformForm(QWidget):
    """Position (mm) and rotation (degrees, about the model's centre) of one model."""

    def __init__(self, transform, on_change, parent=None):
        super().__init__(parent)
        self.transform, self.on_change = transform, on_change
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        self.boxes = {}
        for r, (key, label, unit, limit) in enumerate((("position", "Position", "mm", 1e5),
                                                       ("rotation", "Rotation", "°", 360.0))):
            grid.addWidget(QLabel(f"{label} [{unit}]"), r, 0)
            for c, axis in enumerate("xyz"):
                box = QDoubleSpinBox()
                box.setRange(-limit, limit)
                box.setDecimals(3 if key == "position" else 2)
                box.setSingleStep(1.0 if key == "position" else 5.0)
                box.setPrefix(f"{axis} ")
                box.setValue(float(transform[key][c]))
                box.valueChanged.connect(lambda v, k=key, i=c: self._changed(k, i, v))
                box.setKeyboardTracking(False)
                grid.addWidget(box, r, c + 1)
                self.boxes[key, c] = box
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        grid.addWidget(reset, 2, 3)

    def _changed(self, key, i, value):
        self.transform[key][i] = float(value)
        self.on_change()

    def _reset(self):
        for (key, i), box in self.boxes.items():
            box.blockSignals(True)
            box.setValue(0.0)
            box.blockSignals(False)
            self.transform[key][i] = 0.0
        self.on_change()


class UpdateDialog(QDialog):
    """What updating the neuron from its .mns will change, with the choice for values changed in both places."""

    def __init__(self, name, changes, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Update {name}")
        self.resize(640, 420)
        layout = QVBoxLayout(self)
        text = QLabel(f"<b>{name}</b> was saved with changes since it was imported. Update the full neuron to it?")
        text.setWordWrap(True)
        layout.addWidget(text)
        lines = []
        for key, title in (("source", "Changed in the source (taken)"),
                           ("kept", "Changed here (kept)"),
                           ("conflicts", "Changed in both places")):
            if changes[key]:
                lines += [title + ":"] + [f"   {line}" for line in changes[key]] + [""]
        box = QPlainTextEdit("\n".join(lines).strip() or "Only display settings differ.")
        box.setReadOnly(True)
        layout.addWidget(box, 1)
        self.keep = QCheckBox("Where both changed a value, keep the value set here (untick: take the source's)")
        self.keep.setChecked(True)
        self.keep.setVisible(bool(changes["conflicts"]))
        layout.addWidget(self.keep)
        note = QLabel("Results and the stored characterisation were made with the old version and will be marked "
                      "as outdated.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Update")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class FullNeuronWindow(QMainWindow):
    TITLE = TITLE
    open_source = Signal(str)   # open a component's source file in its own tab
    analyse = Signal(str)       # open the saved .mfn in the Analysis tab

    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.project = FullNeuronProject()
        self.neuron_cad, self.design_cad = CadModel(), CadModel()
        self.design_project = None
        self.surfaces = {NEURON: {}, ACTIVATION: {}}   # untransformed display meshes per model
        self.centers = {NEURON: np.zeros(3), ACTIVATION: np.zeros(3)}
        self.visible = {NEURON: True, ACTIVATION: True}
        self.connections = []          # the design's flow connections (for its tube axis)
        self.valve_mesh = None         # simulation mesh of the valve on its own (cached)
        self.mesh = None               # simulation mesh of the neuron (cached per import)
        self.build, self.linked, self.rows, self.row = None, None, [], None
        self.bodies = None             # the neuron's sheets of the shown rows (shell_bodies / stored shapes)
        self.axes_info = []            # (label, unit) of the swept parameters of the shown rows
        self.axes_targets = []         # [(part, field)] each swept parameter of the shown rows drives
        self.run_axes = []
        self.showing = None            # what the table shows: "solve", "sweep" (a run) or "dataset" (stored)
        self.run_kind = None           # "solve" or "sweep"
        self.outdated = False
        self.selection = None          # ("group", key) | ("neuron", i) | ("design", j) | None
        self.mode = MODEL
        self.worker = None
        self.statuses = {NEURON: None, ACTIVATION: None}   # reference_status of each component
        self._build_ui()
        self._build_actions()
        self._update_actions()
        self.watch = QTimer(self)
        self.watch.timeout.connect(self.check_sources)
        self.watch.start(WATCH_MS)
        self.log_message("Full neuron: File → Import inputs to pre-activation (*.mns) and File → Import pre-activation "
                         "to activation (*.mad), then link them in the Link tab. Help → Quick guide explains the rest.")

    # -----------------------------
    # UI
    # -----------------------------

    def _build_ui(self):
        self.viewport = Viewport(self)
        self.viewport.picked.connect(self._on_pick)
        self.setCentralWidget(self.viewport)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Part", "Role"])
        self.tree.setColumnWidth(0, 190)
        self.tree.itemSelectionChanged.connect(self._tree_selected)
        self.tree.itemChanged.connect(self._tree_checked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        dock = QDockWidget("Model", self)
        dock.setObjectName("full_model_dock")
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.tabs = QTabWidget()
        self.props_area = QScrollArea()
        self.props_area.setWidgetResizable(True)
        self.tabs.addTab(self.props_area, "Properties")
        self.tabs.addTab(self._make_link_tab(), "Link")
        self.tabs.addTab(self._make_run_tab(), "Record && run")
        self.solver_panel = SolverPanel(fields=NEURON_SOLVER_FIELDS)
        self.solver_panel.changed.connect(self._on_solver_changed)
        self.tabs.addTab(self.solver_panel, "Solver")
        self.tabs.addTab(self._make_results_tab(), "Results")
        dock = QDockWidget("Properties", self)
        dock.setObjectName("full_properties_dock")
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(400)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        dock = QDockWidget("Messages", self)
        dock.setObjectName("full_log_dock")
        dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [140], Qt.Vertical)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(lambda: self.worker and self.worker.cancel())
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self._show_properties()

    def _make_link_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()
        self.link_part = QComboBox()
        self.link_part.currentIndexChanged.connect(self._on_link_changed)
        self.link_part.setToolTip("The neuron membrane that is the design's membrane. It is not simulated: the "
                                  "design's pre-simulated response (swept volume against Δp) replaces it.")
        form.addRow("Neuron membrane", self.link_part)
        self.link_driving = QComboBox()
        self.link_driving.currentIndexChanged.connect(self._on_link_changed)
        self.link_driving.setToolTip("The chamber that pushes the design's membrane towards its tube: its pressure "
                                     "minus the other side's is the pre-activation Δp. Automatic: the closed chamber "
                                     "the membrane touches.")
        form.addRow("Driving (pre-activation) chamber", self.link_driving)
        layout.addLayout(form)
        self.link_info = QLabel()
        self.link_info.setWordWrap(True)
        self.link_info.setStyleSheet("color: #555;")
        layout.addWidget(self.link_info)
        layout.addStretch(1)
        return w

    def _make_run_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        box = QGroupBox("Characterisation: parameters to sweep (every combination is solved)")
        v = QVBoxLayout(box)
        self.axes_table = QTableWidget(0, 6)
        self.axes_table.setHorizontalHeaderLabels(["Part", "Parameter [unit]", "Also", "From", "To", "Points"])
        self.axes_table.horizontalHeaderItem(2).setToolTip(
            "Other parts that take the same value at every point (one axis, e.g. both activation membranes' "
            "thickness).")
        self.axes_table.verticalHeader().setVisible(False)
        self.axes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.axes_table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.axes_table.horizontalHeader()
        for c, width in ((2, 44), (3, 70), (4, 70), (5, 48)):  # numbers narrow, the names share the rest
            header.setSectionResizeMode(c, QHeaderView.Fixed)
            self.axes_table.setColumnWidth(c, width)
        for c in (0, 1):
            header.setSectionResizeMode(c, QHeaderView.Stretch)
        self.axes_table.setMinimumHeight(130)
        v.addWidget(self.axes_table, 1)
        row = QHBoxLayout()
        add = QPushButton("Add parameter")
        add.clicked.connect(self._add_axis)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_axis)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        self.points_label = QLabel()
        row.addWidget(self.points_label)
        v.addLayout(row)
        self.store_shapes = QCheckBox("Store the deformed shapes (3D view of every point after reopening; "
                                      "larger file)")
        self.store_shapes.toggled.connect(lambda on: self.project.sweep.__setitem__("store_shapes", bool(on)))
        v.addWidget(self.store_shapes)
        layout.addWidget(box, 1)

        box = QGroupBox("Record (at every point)")
        v = QVBoxLayout(box)
        self.record_list = QListWidget()
        self.record_list.itemChanged.connect(self._record_changed)
        v.addWidget(self.record_list, 1)
        row = QHBoxLayout()
        for text, state in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, on=state: self._record_all(on))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        layout.addWidget(box, 1)

        buttons = QHBoxLayout()
        self.solve_button = QPushButton("Solve once at the set values")
        self.solve_button.clicked.connect(self.solve)
        self.sweep_button = QPushButton("Run characterisation")
        self.sweep_button.clicked.connect(self.run_sweep)
        buttons.addWidget(self.solve_button)
        buttons.addWidget(self.sweep_button)
        layout.addLayout(buttons)
        return w

    def _make_results_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.results_status = QLabel("No results yet.")
        self.results_status.setWordWrap(True)
        layout.addWidget(self.results_status)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._row_selected)
        layout.addWidget(self.table, 1)
        form = QFormLayout()
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.0, 1000.0)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.25)
        self.scale.valueChanged.connect(lambda _: self.mode == RESULTS and self.refresh_view())
        form.addRow("Deformation scale", self.scale)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.show_dataset_button = QPushButton("Show the stored characterisation")
        self.show_dataset_button.clicked.connect(self.show_dataset)
        row.addWidget(self.show_dataset_button)
        self.analyse_button = QPushButton("Analyse…")
        self.analyse_button.setToolTip("Plot the stored characterisation in the Analysis tab")
        self.analyse_button.clicked.connect(self.open_analysis)
        row.addWidget(self.analyse_button)
        export = QPushButton("Export CSV…")
        export.clicked.connect(self.export_csv)
        row.addWidget(export)
        row.addStretch(1)
        layout.addLayout(row)
        return w

    def _action(self, text, slot, shortcut=None, icon=None, tip=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        action.triggered.connect(slot)
        return action

    def _build_actions(self):
        S = QStyle
        self.a_import_neuron = self._action("Import inputs to pre-activation…", self.import_neuron, "Ctrl+I",
                                            S.SP_DialogOpenButton, "Import a neuron project (*.mns)")
        self.a_import_design = self._action("Import pre-activation to activation…", self.import_design, "Ctrl+Shift+I",
                                            S.SP_DialogOpenButton, "Import a simulated design (*.mad)")
        self.a_open = self._action("Open full neuron…", self.open_project, "Ctrl+Shift+O")
        self.a_save = self._action("Save", lambda: self.save_project(False), "Ctrl+S", S.SP_DialogSaveButton)
        self.a_save_as = self._action("Save as…", lambda: self.save_project(True), "Ctrl+Shift+S")
        self.a_solve = self._action("Solve", self.solve, "F5", S.SP_MediaPlay, "Solve once at the set values and record")
        self.a_sweep = self._action("Run characterisation", self.run_sweep, "F6", S.SP_BrowserReload,
                                    "Solve every combination of the swept parameters, record, and keep the dataset "
                                    "with the model")
        self.a_update = self._action("Update", self.update_all, None, S.SP_MessageBoxWarning,
                                     "A component was saved with changes since it was imported: take the new version")
        self.a_analyse = self._action("Analyse characterisation…", self.open_analysis, "F7", S.SP_FileDialogDetailedView,
                                      "Plot the stored characterisation in the Analysis tab")
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
        for a in (self.a_import_neuron, self.a_import_design, self.a_update, None, self.a_open, self.a_save,
                  self.a_save_as):
            m.addSeparator() if a is None else m.addAction(a)
        m = menu.addMenu("&View")
        for a in (*self.view_actions.values(), self.a_reset_cam):
            m.addAction(a)
        m = menu.addMenu("&Simulation")
        m.addAction(self.a_solve)
        m.addAction(self.a_sweep)
        m.addAction(self.a_analyse)
        m = menu.addMenu("&Help")
        m.addAction(self.a_guide)

        tb = QToolBar("Main")
        tb.setObjectName("full_toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_import_neuron, self.a_import_design, self.a_save, None, self.a_solve, self.a_sweep, None,
                  *self.view_actions.values()):
            tb.addSeparator() if a is None else tb.addAction(a)
        tb.insertAction(self.a_solve, self.a_update)
        tb.insertAction(self.view_actions[MODEL], self.a_analyse)
        tb.insertSeparator(self.view_actions[MODEL])
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
        busy = self.worker is not None
        ready = self.project.ready
        for a in (self.a_import_neuron, self.a_import_design, self.a_open):
            a.setEnabled(not busy)
        outdated = [k for k, v in self.statuses.items() if v == OUTDATED]
        self.a_update.setVisible(bool(outdated))
        self.a_update.setEnabled(bool(outdated) and not busy)
        self.a_update.setText("Update" + (f" ({len(outdated)})" if len(outdated) > 1 else ""))
        for a in (self.a_save, self.a_save_as):
            a.setEnabled(ready and not busy)
        for a in (self.a_solve, self.a_sweep):
            a.setEnabled(ready and not busy)
        self.solve_button.setEnabled(ready and not busy)
        self.sweep_button.setEnabled(ready and not busy and bool(self.project.sweep_axes()))
        self.view_actions[RESULTS].setEnabled(self._can_show())
        has_data = bool(self.project.characterisation) and not busy
        self.a_analyse.setEnabled(has_data)
        self.analyse_button.setEnabled(has_data)
        name = Path(self.project.path).name if self.project.path else ""
        self.setWindowTitle(f"{TITLE} — {name}" if name else TITLE)
        parts = []
        if self.project.neuron_source or self.project.neuron:
            parts.append(Path(self.project.neuron_source or self.project.neuron.step_path).name)
        if self.project.design_path:
            parts.append(Path(self.project.design_path).name)
        link = self.project.link.get("part")
        self.status_label.setText(" + ".join(parts) + (f" · linked through {link}" if link else
                                                       (" · not linked" if len(parts) == 2 else ""))
                                  + (" · valve on its own (simulated at every point)" if self.project.valve_only else "")
                                  + (" · results outdated" if self.outdated and self.rows else "")
                                  + "".join(f" · {self._source_name(k)} {v}" for k, v in self.statuses.items()
                                            if v in (OUTDATED, MISSING)))

    def log_message(self, text):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def show_guide(self):
        QMessageBox.information(self, "Quick guide - full neuron", GUIDE)

    # -----------------------------
    # Files
    # -----------------------------

    def _dialog_dir(self):
        return QSettings("MembraneNeuronSimulator", "App").value("last_dir", str(Path.cwd()))

    def _remember_dir(self, path):
        QSettings("MembraneNeuronSimulator", "App").setValue("last_dir", str(Path(path).parent))

    def import_neuron(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Import inputs to pre-activation", self._dialog_dir(),
                                                  "Neuron project (*.mns)")
            if not path:
                return
        self._remember_dir(path)
        try:
            self.project.import_neuron(path)
            self._load_neuron_cad()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, TITLE, f"Could not import the neuron:\n{exc}")
            return
        self.log_message(f"Imported inputs to pre-activation: {Path(path).name} "
                         f"({len(self.project.neuron.parts)} parts).")
        self._after_import()

    def import_design(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Import pre-activation to activation", self._dialog_dir(),
                                                  "Activation function design (*.mad)")
            if not path:
                return
        self._remember_dir(path)
        try:
            self.project.import_design(path)
            self._load_design_cad()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, TITLE, f"Could not import the design:\n{exc}")
            return
        d = self.project.design()
        self.log_message(f"Imported pre-activation to activation: {Path(path).name} (Δp {d.dp_range[0]:.4g} … "
                         f"{d.dp_range[1]:.4g} kPa, outputs {', '.join(d.output_names)}).")
        self._place_design()
        self._after_import()

    def _load_neuron_cad(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            neuron = self.project.neuron
            if not Path(neuron.step_path).exists():
                raise ValueError(f"STEP file not found: {neuron.step_path}")
            bodies = self.neuron_cad.load_step(neuron.step_path)
            neuron.match_bodies(bodies)
            self.surfaces[NEURON] = self.neuron_cad.mesh(mesh_sizes(self.neuron_cad, neuron))
            self.centers[NEURON] = _center(self.surfaces[NEURON])
            self.mesh = None
        finally:
            QApplication.restoreOverrideCursor()

    def _load_design_cad(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            design = self.project.design_project()
            if not Path(design.step_path).exists():
                raise ValueError(f"STEP file not found: {design.step_path}")
            bodies = self.design_cad.load_step(design.step_path)
            design.match_bodies(bodies)
            self.project.design_extents = {p.name: [float(x) for x in b.size] for p, b in zip(design.parts, bodies)}
            self.design_project = design
            self.surfaces[ACTIVATION] = self.design_cad.mesh(mesh_sizes(self.design_cad, design))
            self.centers[ACTIVATION] = _center(self.surfaces[ACTIVATION])
            try:
                self.connections = detect_connections(self.surfaces[ACTIVATION], design.parts, self.design_cad.bodies)
            except Exception:
                self.connections = []
        finally:
            QApplication.restoreOverrideCursor()

    def _place_design(self):
        """Put the design next to the neuron (to its +x side) unless it has been placed already."""
        if not self.surfaces[NEURON] or not self.surfaces[ACTIVATION]:
            return
        t = self.project.transforms[ACTIVATION]
        if any(t["position"]) or any(t["rotation"]):
            return
        n = np.vstack([m.vertices for m in self.surfaces[NEURON].values()])
        d = np.vstack([m.vertices for m in self.surfaces[ACTIVATION].values()])
        gap = 0.15 * max(np.ptp(n, axis=0).max(), np.ptp(d, axis=0).max())
        cn, cd = self.centers[NEURON], self.centers[ACTIVATION]
        t["position"] = [float(n[:, 0].max() + gap - d[:, 0].min()), float(cn[1] - cd[1]), float(cn[2] - cd[2])]

    def _after_import(self):
        self.statuses = {k: self.project.reference_status(k) for k in (NEURON, ACTIVATION)}
        self.build, self.rows, self.row, self.linked = None, [], None, None
        self.bodies, self.axes_info, self.showing = None, [], None
        self.outdated = False
        self.selection = None
        self.viewport.new_model()
        self.valve_mesh = None
        if self.project.neuron is not None:
            self.solver_panel.fields = NEURON_SOLVER_FIELDS
            self.solver_panel.load(self.project.neuron.solver)
        elif self.project.valve_only:   # the valve's own solver settings (it always keeps its contacts)
            self.solver_panel.fields = VALVE_SOLVER_FIELDS
            self.solver_panel.load(self.project.valve_project().solver)
        self._populate_tree()
        self._fill_link()
        self._fill_run()
        self._fill_results()
        self._show_properties()
        self.set_mode(MODEL)
        self._update_actions()

    def open_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open full neuron", self._dialog_dir(), FILTER)
            if not path:
                return
        self._remember_dir(path)
        try:
            project = FullNeuronProject.load(path)
            self.project = project
            if project.neuron is not None:
                self._load_neuron_cad()
            self.surfaces[ACTIVATION] = {}
            self.design_project = None
            if project.design_data is not None:
                self._load_design_cad()
            elif project.design_path:
                raise ValueError(f"The design {project.design_path} was not found.")
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, TITLE, f"Could not open the full neuron:\n{exc}")
            return
        self.log_message(f"Opened {Path(path).name}.")
        self._after_import()
        if project.characterisation:
            self.show_dataset()
            c = project.dataset()
            self.log_message(f"Stored characterisation: {int(c.solved.sum())}/{c.size} point(s)"
                             + ("" if project.dataset_current() else
                                " - the model changed since it was run: run it again"))
        self._announce_statuses()

    def save_project(self, save_as=False):
        path = self.project.path
        if save_as or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save full neuron", str(Path(self._dialog_dir()) /
                                                                             ("full_neuron" + SUFFIX)), FILTER)
            if not path:
                return
        self.project.save(path)
        self._remember_dir(path)
        self.log_message(f"Saved {path}")
        self._update_actions()

    # -----------------------------
    # Model tree
    # -----------------------------

    def _populate_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        link = self.project.link.get("part")
        models = ((NEURON, self.project.neuron.parts if self.project.neuron else [], "neuron"),
                  (ACTIVATION, self.design_project.parts if self.design_project and self.project.design_path else [],
                   "design"))
        for key, parts, kind in models:
            source = self.project.neuron_source if key == NEURON else self.project.design_path
            status = self.statuses.get(key)
            label = Path(source).name if source else "(not imported)"
            group = QTreeWidgetItem([GROUP_NAMES[key], label + (f" — {status}" if status in STATUS_COLORS else "")])
            font = group.font(0)
            font.setBold(True)
            group.setFont(0, font)
            if status in STATUS_COLORS:
                group.setIcon(0, self.style().standardIcon(QStyle.SP_MessageBoxWarning))
                for column in (0, 1):
                    group.setForeground(column, QBrush(QColor(STATUS_COLORS[status])))
            if source:
                tip = f"{source}\n" + {
                    CURRENT: "Up to date with the file.",
                    OUTDATED: "The file was saved with changes since it was imported: right-click → Update.",
                    MISSING: "The file was not found: the copy kept in the full neuron is used."}.get(status, "")
                for column in (0, 1):
                    group.setToolTip(column, tip)
            group.setData(0, Qt.UserRole, ("group", key))
            group.setFlags(group.flags() | Qt.ItemIsUserCheckable)
            group.setCheckState(0, Qt.Checked if self.visible[key] else Qt.Unchecked)
            self.tree.addTopLevelItem(group)
            for i, part in enumerate(parts):
                role = ("Linked: " + ACTIVATION_MEMBRANE) if kind == "neuron" and part.name == link else part.role
                item = QTreeWidgetItem([part.name, role])
                item.setIcon(0, color_icon(ROLE_COLORS[ACTIVATION_MEMBRANE] if role.startswith("Linked")
                                           else part_color(part)))
                item.setData(0, Qt.UserRole, (kind, i))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if part.visible else Qt.Unchecked)
                group.addChild(item)
            if key == ACTIVATION and self.project.valve_only:   # orifices: their area can be set and swept
                for conn in self.project.orifices():
                    item = QTreeWidgetItem([conn, "Orifice"])
                    item.setData(0, Qt.UserRole, ("connection", conn))
                    group.addChild(item)
            group.setExpanded(True)
        self.tree.blockSignals(False)

    def _tree_selected(self):
        items = self.tree.selectedItems()
        self.selection = items[0].data(0, Qt.UserRole) if items else None
        self._show_properties()
        self.tabs.setCurrentIndex(0)
        self.refresh_view()

    def _tree_checked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data is None or column != 0:
            return
        visible = item.checkState(0) == Qt.Checked
        kind, key = data
        if kind == "connection":
            return
        if kind == "group":
            self.visible[key] = visible
        else:
            self._part(kind, key).visible = visible
        self.refresh_view()

    def _part(self, kind, i):
        return (self.project.neuron if kind == "neuron" else self.design_project).parts[i]

    def _select_in_tree(self, data):
        for g in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(g)
            for item in [group] + [group.child(k) for k in range(group.childCount())]:
                if item.data(0, Qt.UserRole) == data:
                    self.tree.setCurrentItem(item)
                    return

    def _on_pick(self, pick, add):
        if pick < 0:
            self.tree.clearSelection()
            return
        self._select_in_tree(("design", pick - DESIGN_BASE) if pick >= DESIGN_BASE else ("neuron", pick))

    # -----------------------------
    # Properties
    # -----------------------------

    def _show_properties(self):
        """Rebuild the Properties tab for the selection (deferred: may be triggered by one of its own widgets)."""
        QTimer.singleShot(0, self._build_properties)

    def _build_properties(self):
        old = self.props_area.takeWidget()
        if old is not None:
            old.deleteLater()
        w = QWidget()
        layout = QVBoxLayout(w)
        sel = self.selection
        title = QLabel()
        title.setStyleSheet("font-weight: 600; font-size: 13px;")
        title.setWordWrap(True)
        layout.addWidget(title)
        info = QLabel()
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        if sel is None:
            title.setText("Select a group or a part in the model tree.")
            info.setText("Click a group (Inputs to pre-activation / Pre-activation to activation) to set its position "
                         "and rotation; click a fluid to change its parameters.")
            layout.addWidget(info)
        elif sel[0] == "group":
            key = sel[1]
            title.setText(GROUP_NAMES[key])
            box = QGroupBox("Placement in the view (about the model's centre)")
            v = QVBoxLayout(box)
            v.addWidget(TransformForm(self.project.transforms[key], self.refresh_view))
            layout.addWidget(box)
            if key == ACTIVATION and self.project.valve_only:
                box = QGroupBox("Valve on its own")
                v = QVBoxLayout(box)
                form = FieldForm()
                form.build([DP_FIELD], {"dp": self.project.valve_value(PREACTIVATION, "dp")})
                form.edited.connect(lambda k, value: self._on_valve_param(PREACTIVATION, k, value))
                v.addWidget(form)
                layout.addWidget(box)
                d = self.project.design()
                info.setText(f"No neuron is imported: the valve is simulated again at every point (with the values set "
                             f"here), so its supply and sink pressures, orifice areas and materials can be changed and "
                             f"swept, together with the Δp across its membrane (Pre-activation Δp). Click a part or an "
                             f"orifice below to change it. Its stored simulation covers Δp {d.dp_range[0]:.4g} … "
                             f"{d.dp_range[1]:.4g} kPa, outputs {', '.join(d.output_names)}.")
            elif key == ACTIVATION and self.project.design_path:
                d = self.project.design()
                info.setText(f"{Path(self.project.design_path).name}: Δp {d.dp_range[0]:.4g} … {d.dp_range[1]:.4g} kPa, "
                             f"outputs {', '.join(d.output_names)}, membrane {d.membrane_area:.4g} mm². The design is "
                             "fixed here; change it in the Pre-activation → activation tab and simulate it again.")
            elif key == NEURON and self.project.neuron is not None:
                info.setText(f"{len(self.project.neuron.parts)} parts. Only fluid parameters and the material of "
                             "membranes and shells can be changed here.")
            layout.addWidget(info)
            self._add_source_box(layout, key)
        elif sel[0] == "connection":
            key = sel[1]
            title.setText(key)
            layout.addWidget(QLabel("Flow connection: <b>orifice</b>"))
            form = FieldForm()
            form.build([AREA_FIELD], {"area": self.project.valve_value(key, "area")})
            form.edited.connect(lambda k, value, n=key: self._on_valve_param(n, k, value))
            layout.addWidget(form)
            law = (self.project.design_data.get("connections") or {}).get(key, {}).get("law", "")
            info.setText(f"Its law Δp = {law} sees A = this area (0: the area of the contact face between the two "
                         "fluids). Add it as a parameter in Record & run to sweep it.")
            layout.addWidget(info)
        else:
            kind, i = sel
            part = self._part(kind, i)
            title.setText(part.name)
            linked = kind == "neuron" and part.name == self.project.link.get("part")
            role = QLabel(f"Role: <b>{part.role}</b>" + (f" &nbsp;·&nbsp; model: <b>{part.props.get('model')}</b>"
                                                         if part.props.get("model") else ""))
            layout.addWidget(role)
            sheet = kind == "neuron" and part.role in DEFORMABLE and not (linked and self.project.design_path)
            if kind == "neuron" and (part.role == CHAMBER or sheet):
                box = QGroupBox("Fluid parameters" if part.role == CHAMBER else "Material")
                v = QVBoxLayout(box)
                form = FieldForm()
                form.build(editable_fields(part), part.props, {"__role__": part.role})
                form.edited.connect(lambda key, value, p=part: self._on_param(p, key, value))
                v.addWidget(form)
                layout.addWidget(box)
                info.setText("The chamber's model is fixed here (change it in the Inputs → pre-activation tab)."
                             if part.role == CHAMBER else
                             f"The material model ({part.props.get('material')}), mesh and supports are fixed here "
                             "(change them in the Inputs → pre-activation tab). The thickness changes the stiffness and "
                             "the contact offset; the mid-surface stays where the CAD puts it.")
            elif linked:
                info.setText(f"Replaced by the pre-simulated membrane of {Path(self.project.design_path).name} "
                             "(see the Link tab).")
            elif kind == "design" and self.project.valve_only and valve_fields(part):
                fields = valve_fields(part, self.project.design_extents.get(part.name))
                box = QGroupBox("Fluid parameters" if part.role == FLUID else
                                "Material and size" if part.role in DEFORMABLE else "Material")
                v = QVBoxLayout(box)
                form = FieldForm()
                values = dict(part.props, **{f.key: self.project.valve_value(part.name, f.key) for f in fields})
                form.build(fields, values, {"__role__": part.role})
                form.edited.connect(lambda k, value, n=part.name: self._on_valve_param(n, k, value))
                v.addWidget(form)
                layout.addWidget(box)
                info.setText("The valve is simulated again at every point with this value (its kinds of parts, mesh "
                             "and connections are fixed here: change those in the Pre-activation → activation tab).")
            elif kind == "design":
                text = "Part of the design: fixed by its pre-simulation (change it in the Pre-activation → activation " \
                       "tab and simulate it again)."
                if part.role == FLUID and part.props.get("model") == "Constant pressure":
                    text = f"Pressure {part.props.get('pressure', 0.0)} kPa. " + text
                if part.role == FLUID and part.props.get("outputs"):
                    text += " Outputs: " + ", ".join(f"{o['name']} (segment {o['segment']})"
                                                      for o in part.props["outputs"])
                info.setText(text)
            else:
                info.setText("Only fluid parameters and the material of membranes and shells can be changed in the "
                             "full neuron.")
            layout.addWidget(info)
        layout.addStretch(1)
        self.props_area.setWidget(w)

    def _on_param(self, part, key, value):
        if part.props.get(key) == value:
            return
        part.props[key] = value
        self.outdated = True
        self.log_message(f"{part.name}: {key} = {value}")
        self._fill_results()
        self._update_actions()

    def _on_valve_param(self, name, key, value):
        if self.project.valve_value(name, key) == value:
            return
        self.project.set_valve_value(name, key, value)
        self.outdated = True
        self.log_message(f"{name}: {key} = {value}")
        self._fill_results()
        self._update_actions()

    def _on_solver_changed(self, key, value):
        if self.project.valve_only:
            self.project.set_valve_value(SOLVER_KEY, key, value)
            self.outdated = True
        elif self.project.neuron is not None:
            setattr(self.project.neuron.solver, key, value)
            self.outdated = True
            if key == "contact":   # the penetration record comes and goes with the contact switch
                if self.project.record is not None and "penetration" not in self.project.record:
                    self.project.record.append("penetration")
                QTimer.singleShot(0, self._fill_run)

    # -----------------------------
    # Link
    # -----------------------------

    def _fill_link(self):
        for combo in (self.link_part, self.link_driving):
            combo.blockSignals(True)
            combo.clear()
        self.link_part.addItem("(not linked)", None)
        for name in self.project.link_candidates():
            self.link_part.addItem(name, name)
        self.link_driving.addItem(AUTOMATIC, AUTOMATIC)
        for p in self.project.chambers():
            self.link_driving.addItem(p.name, p.name)
        k = self.link_part.findData(self.project.link.get("part"))
        self.link_part.setCurrentIndex(max(k, 0))
        k = self.link_driving.findData(self.project.link.get("driving") or AUTOMATIC)
        self.link_driving.setCurrentIndex(max(k, 0))
        for combo in (self.link_part, self.link_driving):
            combo.blockSignals(False)
        ok = self.project.neuron is not None and bool(self.project.design_path)
        self.link_part.setEnabled(ok)
        self.link_driving.setEnabled(ok)
        if self.project.valve_only:
            self.link_info.setText("Valve on its own: nothing to link. Import an inputs-to-pre-activation project to "
                                   "use the design as a part of a neuron instead.")
        elif not ok:
            self.link_info.setText("Import both an inputs-to-pre-activation project and a pre-activation-to-activation "
                                   "design to link them.")
        elif not self.project.link.get("part"):
            self.link_info.setText("Choose the neuron membrane that is the design's membrane.")
        else:
            self.link_info.setText(f"{self.project.link['part']} is replaced by the pre-simulated membrane of "
                                   f"{Path(self.project.design_path).name}; the pre-activation Δp across it drives the "
                                   "design's outputs. The link is drawn as a line between the two models.")

    def _on_link_changed(self, *_):
        part, driving = self.link_part.currentData(), self.link_driving.currentData() or AUTOMATIC
        if part == self.project.link.get("part") and driving == self.project.link.get("driving"):
            return
        self.project.link = {"part": part, "driving": driving}
        self.project.record = None if self.project.record is None else self.project.record
        self.outdated = True
        self.log_message(f"Link: {part or 'none'} (driving chamber: {driving}).")
        self._populate_tree()
        self._fill_link()
        self._fill_run()
        self._update_actions()
        self.refresh_view()

    # -----------------------------
    # Record & run
    # -----------------------------

    def _fill_run(self):
        self._fill_axes()
        self.store_shapes.blockSignals(True)
        self.store_shapes.setChecked(bool(self.project.sweep.get("store_shapes", True)))
        self.store_shapes.blockSignals(False)
        self.record_list.blockSignals(True)
        self.record_list.clear()
        chosen = set(self.project.recorded()) if self.project.ready else set()
        for key, label, unit in (self.project.catalogue() if self.project.ready else []):
            item = QListWidgetItem(f"{label} [{unit}]")
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in chosen else Qt.Unchecked)
            self.record_list.addItem(item)
        self.record_list.blockSignals(False)

    # the sweep axes: one table row per parameter, its widgets write straight into project.sweep["axes"]

    def _parameters_by_part(self):
        out = {}
        for part, field, label, unit in self.project.parameters():
            out.setdefault(part, []).append((field, label, unit))
        return out

    def _fill_axes(self):
        params = self._parameters_by_part()
        axes = self.project.sweep["axes"]
        self.axes_table.setRowCount(0)
        self.axes_table.setRowCount(len(axes))
        for r, axis in enumerate(axes):
            part = QComboBox()
            for name in params:
                part.addItem(name, name)
            if axis.get("part") not in params and params:
                axis["part"] = next(iter(params))
            part.setCurrentIndex(max(part.findData(axis.get("part")), 0))
            field = QComboBox()
            for combo in (part, field):
                combo.setMinimumContentsLength(4)
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self._fill_fields(field, params.get(axis.get("part"), []), axis)
            also = self._also_button(axis)
            lo, hi = QDoubleSpinBox(), QDoubleSpinBox()
            for s, key in ((lo, "from"), (hi, "to")):
                s.setRange(-1e9, 1e9)
                s.setDecimals(4)
                s.setButtonSymbols(QDoubleSpinBox.NoButtons)
                s.setKeyboardTracking(False)
                s.setValue(float(axis.get(key, 0.0)))
            n = QSpinBox()
            n.setRange(1, 10000)
            n.setValue(int(axis.get("points", 5)))
            n.setButtonSymbols(QSpinBox.NoButtons)
            for c, widget in enumerate((part, field, also, lo, hi, n)):
                self.axes_table.setCellWidget(r, c, widget)
            part.currentIndexChanged.connect(lambda _, a=axis, c=part: self._axis_part(a, c))
            field.currentIndexChanged.connect(lambda _, a=axis, f=field: self._axis_field(a, f))
            lo.valueChanged.connect(lambda x, a=axis: self._axis_value(a, "from", x))
            hi.valueChanged.connect(lambda x, a=axis: self._axis_value(a, "to", x))
            n.valueChanged.connect(lambda x, a=axis: self._axis_value(a, "points", int(x)))
        self._update_points()

    def _fill_fields(self, combo, fields, axis):
        combo.blockSignals(True)
        combo.clear()
        for key, label, unit in fields:
            name = label.split(" ", 1)[1] if " " in label else label
            combo.addItem(f"{name} [{unit}]" if unit else name, (key, unit))
        keys = [k for k, _, _ in fields]
        if axis.get("field") not in keys and keys:
            axis["field"] = "pressure" if "pressure" in keys else keys[0]
            self._default_range(axis)
        combo.setCurrentIndex(keys.index(axis["field"]) if axis.get("field") in keys else -1)
        combo.setToolTip(combo.currentText())
        combo.blockSignals(False)

    def _also_button(self, axis):
        """The "Also" cell: tick other parts with the same parameter to drive them with this axis too."""
        candidates = self.project.also_candidates(axis)
        axis["also"] = [p for p in axis.get("also") or () if p in candidates]
        button = QToolButton()
        button.setPopupMode(QToolButton.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setEnabled(bool(candidates))
        menu = QMenu(button)
        for name in candidates:
            action = menu.addAction(self.project.also_label(name, axis.get("field")))
            action.setCheckable(True)
            action.setChecked(name in axis["also"])
            action.toggled.connect(lambda on, a=axis, n=name, b=button: self._axis_also(a, n, on, b))
        button.setMenu(menu)
        self._also_text(axis, button)
        return button

    def _also_text(self, axis, button):
        also = axis.get("also") or []
        button.setText(f"+{len(also)}" if also else "—")
        button.setToolTip("Also drives: " + ", ".join(self.project.also_label(n, axis.get("field")) for n in also)
                          if also else "Drive the same parameter of other parts with this axis (they take the same "
                          "value); for a membrane's size, also its size in the other direction.")

    def _axis_also(self, axis, name, on, button):
        also = [p for p in axis.get("also") or () if p != name]
        axis["also"] = also + [name] if on else also
        self._also_text(axis, button)
        self._update_points()

    def _default_range(self, axis):
        """A starting range for a newly chosen material parameter: half to twice its current value."""
        if self.project.valve_only:
            if axis.get("field") == "dp":
                axis["from"], axis["to"] = self.project.design().dp_range
                return
            if axis.get("field") in SIZE_FIELDS:   # a membrane's size: 80 … 120 % of the CAD
                value = float(self.project.valve_value(axis.get("part"), axis["field"]) or 0.0)
                if value > 0:
                    axis["from"], axis["to"], axis["points"] = 0.8 * value, 1.2 * value, 3
                return
            if axis.get("field") in MOVE_FIELDS:   # a rigid body's displacement: ±0.5 mm about its set-up move
                value = float(self.project.valve_value(axis.get("part"), axis["field"]) or 0.0)
                axis["from"], axis["to"], axis["points"] = value - 0.5, value + 0.5, 3
                return
            if axis.get("field") not in SHEET_FIELDS + SOLID_FIELDS + ("area",):
                return
            value = float(self.project.valve_value(axis.get("part"), axis["field"]) or 0.0)
        else:
            if axis.get("field") not in SHEET_FIELDS:
                return
            part = self.project.part(axis.get("part"))
            value = float((part.props.get(axis["field"]) if part else 0.0) or 0.0)
        if value > 0 and axis.get("field") != "poisson_ratio":
            axis["from"], axis["to"] = 0.5 * value, 2.0 * value

    def _axis_part(self, axis, combo):
        axis["part"] = combo.currentData()
        QTimer.singleShot(0, self._fill_axes)   # the field list and "Also" depend on the part

    def _axis_field(self, axis, combo):
        data = combo.currentData()
        if data and data[0] != axis.get("field"):
            axis["field"] = data[0]
            self._default_range(axis)
            QTimer.singleShot(0, self._fill_axes)

    def _axis_value(self, axis, key, value):
        axis[key] = value
        self._update_points()

    def _add_axis(self):
        params = self._parameters_by_part()
        if not params:
            QMessageBox.information(self, TITLE, "Import an inputs-to-pre-activation project with chambers (or only a "
                                                 "pre-activation-to-activation design) first.")
            return
        used = {target(n, a.get("field")) for a in self.project.sweep["axes"] for n in axis_parts(a)}
        inputs = [PREACTIVATION] + self.project.orifices() if self.project.valve_only else self.project.inputs()
        free = [(p, f) for p in inputs + [p for p in params if p not in inputs] for f, _, _ in params.get(p, [])
                if (p, f) not in used]
        part, field = free[0] if free else (next(iter(params)), params[next(iter(params))][0][0])
        axis = default_axis(part, field)
        self._default_range(axis)
        self.project.sweep["axes"].append(axis)
        self._fill_axes()

    def _remove_axis(self):
        rows = self.axes_table.selectionModel().selectedRows()
        r = rows[0].row() if rows else self.axes_table.currentRow()
        axes = self.project.sweep["axes"]
        if r < 0 and axes:
            r = len(axes) - 1
        if 0 <= r < len(axes):
            del axes[r]
            self._fill_axes()

    def _update_points(self):
        axes = self.project.sweep_axes()
        n = int(np.prod([len(v) for _, _, v in axes])) if axes else 0
        repeated = len(axes) < len(self.project.sweep["axes"])
        self.points_label.setText((f"{n} point(s)" if axes else "No parameters: add one")
                                  + (" · a parameter is listed twice (used once)" if repeated else ""))
        self._update_actions()

    def _record_changed(self, _):
        self.project.record = [self.record_list.item(k).data(Qt.UserRole) for k in range(self.record_list.count())
                               if self.record_list.item(k).checkState() == Qt.Checked]
        self._fill_results()

    def _record_all(self, on):
        self.record_list.blockSignals(True)
        for k in range(self.record_list.count()):
            self.record_list.item(k).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.record_list.blockSignals(False)
        self._record_changed(None)

    def _prepare(self):
        """The linked neuron and its simulation mesh (meshed on the main thread, cached)."""
        linked = self.project.linked_project()
        if self.mesh is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.log_message("Meshing the neuron…")
                self.mesh = generate_mesh(self.neuron_cad, linked)
            finally:
                QApplication.restoreOverrideCursor()
        return linked

    def solve(self):
        self._run("solve", [])

    def run_sweep(self):
        axes = self.project.sweep_axes()
        if not axes:
            QMessageBox.information(self, TITLE, "Add at least one parameter to sweep (Record && run tab).")
            return
        if not self.project.recorded():
            QMessageBox.information(self, TITLE, "Tick at least one quantity to record.")
            return
        n = int(np.prod([len(v) for _, _, v in axes]))
        if n > 500 and QMessageBox.question(
                self, TITLE, f"The characterisation has {n} points (every combination of the parameters). "
                             "Run it?") != QMessageBox.Yes:
            return
        self._run("sweep", axes)

    def _run(self, kind, axes):
        if self.worker is not None or not self.project.ready:
            return
        if self.project.valve_only:
            return self._run_valve(kind, axes)
        try:
            linked = self._prepare()
        except Exception as exc:
            QMessageBox.warning(self, TITLE, str(exc))
            return
        self.linked, self.rows, self.row, self.run_kind, self.build = linked, [], None, kind, None
        self.run_axes = axes
        self.axes_info = [self.project.axis_label(p, f) for p, f, _ in axes]
        self.axes_targets = [[target(n, f) for n in parts_of(p)] for p, f, _ in axes]
        self.bodies = None
        self.showing = kind
        self.run_started = time.time()
        self._fill_results()
        self.worker = Worker(run_grid, self.neuron_cad, linked, self.mesh, axes, parent=self)
        self.worker.item.connect(self._add_row)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self.log_message)
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.log_message("Solving the full neuron…" if kind == "solve" else
                         f"Characterising: {int(np.prod([len(v) for _, _, v in axes]))} point(s) over "
                         + ", ".join(label for label, _ in self.axes_info) + "…")
        self._update_actions()
        self.tabs.setCurrentIndex(4)
        self.worker.start()

    def _run_valve(self, kind, axes):
        """The valve on its own: simulated again at every point (meshed on the main thread, cached)."""
        try:
            valve = self.project.valve_project()
            if self.valve_mesh is None:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    self.log_message("Meshing the valve…")
                    self.valve_mesh = generate_activation_mesh(self.design_cad, valve)
                finally:
                    QApplication.restoreOverrideCursor()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, TITLE, str(exc))
            return
        self.linked, self.rows, self.row, self.run_kind, self.build = valve, [], None, kind, None
        self.run_axes = axes
        self.axes_info = [self.project.axis_label(p, f) for p, f, _ in axes]
        self.axes_targets = [[target(n, f) for n in parts_of(p)] for p, f, _ in axes]
        self.bodies = None
        self.showing = kind
        self.run_started = time.time()
        self._fill_results()
        self.worker = Worker(run_valve_grid, self.design_cad, valve, self.valve_mesh, axes,
                             float(self.project.valve_value(PREACTIVATION, "dp")), parent=self)
        self.worker.item.connect(self._add_row)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self.log_message)
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.log_message("Simulating the valve…" if kind == "solve" else
                         f"Characterising the valve: {int(np.prod([len(v) for _, _, v in axes]))} point(s) over "
                         + ", ".join(label for label, _ in self.axes_info) + " (simulated at every point)…")
        self._update_actions()
        self.tabs.setCurrentIndex(4)
        self.worker.start()

    def _on_progress(self, fraction, text):
        self.progress.setValue(int(100 * max(fraction, 0.0)))
        self.status_label.setText(text)

    def _where(self, row):
        return ", ".join(f"{label} = {v:.4g} {unit}".rstrip() for (label, unit), v in zip(self.axes_info, row["params"]))

    def _add_row(self, row):
        self.rows.append(row)
        values = row["values"]
        recorded = ", ".join(f"{self._label(k)} = {values_or_nan(row, k):.4g}" for k in self.project.recorded())
        where = self._where(row)
        self.log_message(f"  {where + ': ' if where else ''}{recorded}" + ("" if row["converged"] else
                                                                          "  (NOT converged)")
                         + ("  ⚠ EXTRAPOLATING" if values.get("warnings") else ""))
        self._fill_results()

    def _done(self, build):
        self.worker = None
        self.build = build
        self.bodies = ((getattr(build, "display", None) or display_bodies(build, self.linked))
                       if self.project.valve_only else shell_bodies(build, self.linked.parts))
        self.outdated = False
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        ok = sum(r["converged"] for r in self.rows)
        self.log_message(f"Done: {ok}/{len(self.rows)} point(s) converged.")
        if self.run_kind == "sweep":
            self._store_dataset(self.bodies)
        self.row = len(self.rows) - 1 if self.rows else None
        self._fill_results()
        self._update_actions()
        self.set_mode(RESULTS)
        outside = [r for r in self.rows if r["values"].get("warnings")]
        if outside:
            lo, hi = self.project.design().dp_range
            dps = [r["values"]["dp"] for r in outside]
            QMessageBox.warning(self, TITLE, (outside[0]["values"]["warnings"][0] if len(self.rows) == 1 else
                                              f"{len(outside)} of {len(self.rows)} point(s) put the pre-activation Δp "
                                              f"outside the design's simulated range ({lo:.4g} … {hi:.4g} kPa; here "
                                              f"{min(dps):.4g} … {max(dps):.4g} kPa): there the design is "
                                              "EXTRAPOLATED, not interpolated (orange rows in the Results tab)."))

    def _store_dataset(self, bodies):
        """Keep the characterisation with the model (and save the .mfn when it has a file)."""
        self.project.characterisation = make_dataset(self.project, self.run_axes, self.rows, bodies,
                                                     self.project.sweep.get("store_shapes", True),
                                                     time.time() - self.run_started)
        c = self.project.dataset()
        done = int(c.solved.sum())
        self.log_message(f"Characterisation: {done}/{c.size} point(s), {len(c.keys)} recorded quantities"
                         + ("" if c.complete else " (INCOMPLETE: stopped early)") + ".")
        if self.project.path:
            self.project.save(self.project.path)
            self.log_message(f"Saved with the model in {self.project.path}")
        else:
            self.log_message("Save the full neuron (Ctrl+S) to keep the characterisation with the model.")
            self.save_project(False)

    def _failed(self, text):
        self.worker = None
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.log_message("Stopped: " + text)
        if self.run_kind == "sweep" and self.rows:
            self._store_dataset(None)  # what was solved (no shapes: the build did not come back)
            self.bodies = []
        if text != "Cancelled.":
            QMessageBox.warning(self, TITLE, text.split("\n\n")[0])
        self._fill_results()
        self._update_actions()

    def show_dataset(self):
        """Show the stored characterisation in the table and the 3D view."""
        c = self.project.dataset()
        if c is None:
            self.rows, self.row, self.axes_info, self.bodies, self.showing = [], None, [], None, None
        else:
            shapes = c.shapes()
            self.rows = c.rows(shapes)
            self.row = 0 if self.rows else None
            self.axes_info = [(a["label"], a["unit"]) for a in c.axes]
            self.axes_targets = [[target(n, a["field"]) for n in axis_parts(a)] for a in c.axes]
            self.bodies = shapes or []
            self.showing = "dataset"
        self._fill_results()
        self._update_actions()
        if self.rows:
            self.set_mode(RESULTS)

    # -----------------------------
    # Results
    # -----------------------------

    def _label(self, key):
        return next((label for k, label, _ in self.project.catalogue() if k == key), key)

    def _unit(self, key):
        return next((unit for k, _, unit in self.project.catalogue() if k == key), "")

    def _keys(self):
        """The columns of the table: what the shown rows recorded."""
        if self.showing == "dataset" and self.project.characterisation:
            return [k["key"] for k in self.project.characterisation["keys"]]
        return self.project.recorded() if self.project.ready else []

    def _fill_results(self):
        keys = self._keys()
        headers = [f"{label} [{unit}]" for label, unit in self.axes_info] + \
                  [f"{self._label(k)} [{self._unit(k)}]" for k in keys] + ["Converged"]
        self.table.blockSignals(True)
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            values = list(row["params"]) + [values_or_nan(row, k) for k in keys]
            warning = "\n".join(row["values"].get("warnings") or [])
            for c, v in enumerate(values):
                item = QTableWidgetItem(fmt(v) if v is not None else "")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if warning:  # outside the design's simulated range: extrapolated
                    item.setBackground(QColor("#ffe0b2"))
                    item.setToolTip(warning)
                self.table.setItem(r, c, item)
            status = QTableWidgetItem("held (tube shut)" if row.get("held") else
                                      ("yes" if row["converged"] else "NO") + (" · EXTRAPOLATING" if warning else ""))
            if warning:
                status.setBackground(QColor("#ffe0b2"))
                status.setToolTip(warning)
            self.table.setItem(r, len(values), status)
        if self.row is not None and self.row < len(self.rows):
            self.table.selectRow(self.row)
        self.table.blockSignals(False)
        self.results_status.setText(self._status_text())
        self.show_dataset_button.setEnabled(bool(self.project.characterisation) and self.worker is None
                                            and self.showing != "dataset")

    def _status_text(self):
        lines = []
        c = self.project.dataset()
        if c is not None:
            axes = ", ".join(f"{a['label']} {a['values'].min():.4g} … {a['values'].max():.4g} {a['unit']} "
                             f"({len(a['values'])})" for a in c.axes)
            current = self.project.dataset_current()
            lines.append(f"<b>Stored characterisation</b> ({self.project.characterisation.get('created', '')}): "
                         f"{int(c.solved.sum())}/{c.size} point(s) over {axes or 'no parameters'}; "
                         f"{len(c.keys)} recorded quantities"
                         + ("" if c.complete else " · <b>incomplete</b>")
                         + (f" · {int((c.solved & ~c.converged).sum())} not converged"
                            if (c.solved & ~c.converged).any() else "")
                         + (f" · {int(c.extrapolated.sum())} extrapolating" if c.extrapolated.any() else "")
                         + ("" if current else " · <span style='color:#c62828'><b>the model changed since: run the "
                                               "characterisation again</b></span>")
                         + (" · saved in the .mfn" if self.project.path else " · <b>not saved yet</b>"))
        else:
            lines.append("No characterisation yet: choose the parameters to sweep and what to record in the "
                         "Record &amp; run tab, then Run characterisation (F6).")
        n = len(self.rows)
        if n:
            what = {"dataset": "Showing the stored characterisation", "sweep": "Characterisation run",
                    "solve": "Single solve at the set values"}.get(self.showing, "Results")
            outside = sum(bool(r["values"].get("warnings")) for r in self.rows)
            lines.append(f"{what}: {sum(bool(r['converged']) for r in self.rows)}/{n} point(s) converged"
                         + (f" · ⚠ {outside} outside the design's simulated Δp range: EXTRAPOLATING (orange)"
                            if outside else "")
                         + (" · the model changed since" if self.outdated and self.showing != "dataset" else "")
                         + ". Click a row to show that point in 3D.")
        return "<br>".join(lines)

    def _row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.row = rows[0].row()
            if self._can_show():
                self.set_mode(RESULTS)

    def export_csv(self):
        if not self.rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", str(Path(self._dialog_dir()) / "full_neuron.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        keys = self._keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"{label} [{unit}]" for label, unit in self.axes_info]
                            + [f"{self._label(k)} [{self._unit(k)}]" for k in keys] + ["converged", "extrapolating"])
            for row in self.rows:
                writer.writerow(list(row["params"]) + [values_or_nan(row, k) for k in keys]
                                + [row["converged"], bool(row["values"].get("warnings"))])
        self.log_message(f"Exported {path}")

    # -----------------------------
    # View
    # -----------------------------

    def _can_show(self):
        return bool(self.rows) and self.bodies is not None

    def set_mode(self, mode):
        if mode == RESULTS and not self._can_show():
            mode = MODEL
        self.mode = mode
        self.view_actions[mode].setChecked(True)
        self.view_actions[MODEL if mode == RESULTS else RESULTS].setChecked(False)
        self.refresh_view()

    def _on_transparency(self, value):
        self.viewport.opacity = 1.0 - value / 100.0
        self.refresh_view()

    def _placed(self, key, points):
        t = self.project.transforms[key]
        return transform_points(points, self.centers[key], t["position"], t["rotation"])

    def _design_move(self, part, row=None):
        """Where a design rigid body is drawn relative to its CAD position (mm): its set-up move as set here, or at
        the shown point when the rows sweep it."""
        if part.role != RIGID:
            return np.zeros(3)
        if self.project.valve_only:
            move = np.array([float(self.project.valve_value(part.name, k) or 0.0) for k in MOVE_FIELDS])
        else:
            move = np.array([float(part.props.get(k) or 0.0) for k in MOVE_FIELDS])
        if row is not None and self.showing:
            for drives, value in zip(self.axes_targets, row.get("params") or ()):
                for name, field in drives:
                    if name == part.name and field in MOVE_FIELDS and value is not None:
                        move[MOVE_FIELDS.index(field)] = float(value)
        return move

    def _scene(self, row=None):
        """(surfaces, parts) of both models in one dict each, placed, keyed by pick id; design rigid bodies at their
        set-up move (at `row`'s, when it sweeps one)."""
        surfaces, parts = {}, {}
        link = self.project.link.get("part")
        if self.visible[NEURON] and self.project.neuron is not None:
            for i, m in self.surfaces[NEURON].items():
                part = self.project.neuron.parts[i]
                if part.name == link:  # shown as the design's membrane
                    part = PartSettings(part.name, ACTIVATION_MEMBRANE, {}, part.visible)
                surfaces[i] = SimpleNamespace(vertices=self._placed(NEURON, m.vertices), faces=m.faces)
                parts[i] = part
        if self.visible[ACTIVATION] and self.design_project is not None and self.project.design_path:
            for j, m in self.surfaces[ACTIVATION].items():
                part = self.design_project.parts[j]
                surfaces[DESIGN_BASE + j] = SimpleNamespace(
                    vertices=self._placed(ACTIVATION, m.vertices + self._design_move(part, row)), faces=m.faces)
                parts[DESIGN_BASE + j] = part
        return surfaces, parts

    def _selected_ids(self):
        sel = self.selection
        if sel is None:
            return []
        if sel[0] == "connection":
            return []
        if sel[0] == "group":
            ids = self.surfaces[sel[1]].keys()
            return [i if sel[1] == NEURON else DESIGN_BASE + i for i in ids]
        return [sel[1] if sel[0] == "neuron" else DESIGN_BASE + sel[1]]

    def refresh_view(self):
        showing = self.mode == RESULTS and self._can_show() and self.row is not None and self.row < len(self.rows)
        surfaces, parts = self._scene(self.rows[self.row] if showing else None)
        if not surfaces:
            self.viewport._clear()
            self.viewport.plotter.render()
            return
        self.viewport.set_bounds(surfaces)
        if showing:
            self._show_results(surfaces, parts)
        else:
            self.viewport.show_model(surfaces, parts, self._selected_ids())
        self._show_outputs()
        self._show_link()
        self.viewport.plotter.render()

    def _show_results(self, surfaces, parts):
        """The neuron's sheets and the design deformed at the selected point, both coloured by displacement (one
        scale), everything else as context."""
        vp = self.viewport
        vp._clear()
        row = self.rows[self.row]
        scale = self.scale.value()
        bodies = self.bodies or []
        coords = row.get("coords") if bodies else None
        sheets = []   # (name, frame) of the neuron
        if coords is not None:
            sheets = [(b["name"], dict(b, x=np.asarray(x, float))) for b, x in zip(bodies, coords)
                      if np.all(np.isfinite(x))]
        design = self.project.design() if self.project.design_path else None
        dp = row["values"].get("dp")
        frames = design.frames.at(dp) if design is not None and design.frames is not None and dp is not None \
            and np.isfinite(dp) else []
        if self.project.valve_only:   # the valve simulated at this point: its own bodies are the frames
            frames, sheets = [f for _, f in sheets], []
        by_name ={p.name: p for p in self.design_project.parts} if self.design_project is not None else {}
        neuron_parts = self.project.neuron.parts if self.project.neuron is not None else []
        neuron_names = {p.name: p for p in neuron_parts}
        shown_sheets = [(n, f) for n, f in sheets if self.visible[NEURON] and neuron_names.get(n) is not None
                        and neuron_names[n].visible]
        shown_frames = [f for f in frames if self.visible[ACTIVATION]
                        and (by_name.get(f["name"]) is None or by_name[f["name"]].visible)]
        top = max([float(np.linalg.norm(f["x"] - f["rest"], axis=1).max()) for _, f in shown_sheets]
                  + [float(np.linalg.norm(f["x"] - f["rest"], axis=1).max()) for f in shown_frames] + [1e-12])
        drawn = {n for n, _ in sheets}
        deformed = {f["name"] for f in frames}
        for pick, m in surfaces.items():
            part = parts[pick]
            if pick < DESIGN_BASE and part.name in drawn:
                continue
            if pick >= DESIGN_BASE and part.name in deformed:  # drawn deformed below
                continue
            opacity = 0.9 if pick >= DESIGN_BASE and part.role in DEFORMABLE + (RIGID,) else 0.25
            if pick < DESIGN_BASE and part.role == CHAMBER:
                opacity = 0.08
            if pick < DESIGN_BASE and part.role == ACTIVATION_MEMBRANE:  # the linked membrane (not simulated)
                opacity = 0.35
            vp._add(f"body{pick}", polydata(m.vertices, m.faces), body=pick, color=part_color(part),
                    opacity=min(opacity, max(vp.opacity, 0.08)) if part.role != ACTIVATION_MEMBRANE else opacity,
                    pickable=True)
        bar = dict(title="|u| [mm]", vertical=True, position_x=0.86, position_y=0.1, height=0.7, width=0.06,
                   fmt="%.3g", color="black")
        first = True
        index = {p.name: i for i, p in enumerate(neuron_parts)}
        for k, (name, f) in enumerate(shown_sheets):
            pd = _frame_mesh(f, scale, lambda x: self._placed(NEURON, x))
            vp._add(f"result{k}", pd, body=index.get(name), scalars="|u| [mm]", cmap="turbo", clim=(0.0, top),
                    show_scalar_bar=first, scalar_bar_args=bar, pickable=True)
            first = False
        for n, f in enumerate(shown_frames):  # the design at this point's pre-activation (stored FEM solution)
            pd = _frame_mesh(f, scale, lambda x: self._placed(ACTIVATION, x))
            vp._add(f"design_frame{n}", pd, scalars="|u| [mm]", cmap="turbo", clim=(0.0, top),
                    show_scalar_bar=first, scalar_bar_args=bar, show_edges=False, pickable=False)
            first = False
        text = [f"{self._label(k)}: {values_or_nan(row, k):.4g} {self._unit(k)}" for k in self._keys()]
        warnings = row["values"].get("warnings") or []
        notes = []
        if warnings and design is not None:
            lo, hi = design.dp_range
            vp.plotter.add_text(f"EXTRAPOLATING: pre-activation Δp = {dp:.4g} kPa is outside the design's simulated "
                                f"{lo:.4g} … {hi:.4g} kPa", position="lower_left", font_size=10, color="#c62828",
                                name="extrapolating")
            vp._names.append("extrapolating")
        elif design is not None and design.frames is None:
            notes.append("The design was simulated before its FEM solution was stored: shown undeformed "
                         "(simulate it again to see it deform)")
        if coords is None:
            notes.append("The deformed neuron was not stored with this characterisation: shown undeformed")
        if notes and not warnings:
            vp.plotter.add_text("\n".join(notes), position="lower_left", font_size=9, color="#555555",
                                name="no_frames")
            vp._names.append("no_frames")
        where = self._where(row)
        if where:
            text.insert(0, where)
        vp.plotter.add_text("\n".join(text), position="upper_left", font_size=9, color="black", name="recorded")
        vp._names.append("recorded")
        vp._finish()

    def _show_outputs(self):
        """The design's output segments, coloured and named (with their values in the results view)."""
        if self.design_project is None or not self.visible[ACTIVATION] or not self.project.design_path:
            return
        parts = self.design_project.parts
        connections = [(c, self.design_project.connection(c.key, outside=c.b is None)) for c in self.connections]
        row = self.rows[self.row] if self.mode == RESULTS and self.row is not None and self.rows else None
        labels = []
        for j, part in enumerate(parts):
            outputs = part.props.get("outputs") or ()
            if part.role != FLUID or not outputs or not part.visible or j not in self.surfaces[ACTIVATION]:
                continue
            surface = self.surfaces[ACTIVATION][j]
            try:
                axis, _ = channel_axis(surface, j, connections, parts)
            except Exception:
                continue
            segments = max(int(part.props.get("segments", 10)), 1)
            placed = SimpleNamespace(vertices=self._placed(ACTIVATION, surface.vertices), faces=surface.faces)
            for n, o in enumerate(outputs):
                faces = segment_faces(surface, axis, segments, min(int(o["segment"]), segments))
                if not len(faces):
                    continue
                self.viewport._add(f"output{j}_{n}", _raised(placed, faces), color=OUTPUT_COLORS[n % len(OUTPUT_COLORS)],
                                   opacity=0.85, show_edges=False, pickable=False)
                value = values_or_nan(row, f"out:{o['name']}") if row is not None else None
                text = o["name"] if value is None else f"{o['name']}: {value:.4g} kPa"
                labels.append((placed.vertices[placed.faces[faces]].mean(axis=(0, 1)), text))
        if labels:
            self.viewport.plotter.add_point_labels(np.array([x for x, _ in labels]), [t for _, t in labels],
                                                   name="output_names", font_size=12, point_size=1, shape_opacity=0.6,
                                                   always_visible=True, show_points=False, render=False)
            self.viewport._names.append("output_names")

    def _show_link(self):
        """A line from the linked neuron membrane to the design's membrane."""
        link = self.project.link.get("part")
        if not link or self.design_project is None or not all(self.visible.values()) or not self.project.neuron:
            return
        i = next((k for k, p in enumerate(self.project.neuron.parts) if p.name == link), None)
        membranes = [j for j, p in enumerate(self.design_project.parts) if p.role in DEFORMABLE]
        if i not in self.surfaces[NEURON] or not membranes:
            return
        a = self._placed(NEURON, self.surfaces[NEURON][i].vertices).mean(axis=0)
        b = np.vstack([self._placed(ACTIVATION, self.surfaces[ACTIVATION][j].vertices) for j in membranes
                       if j in self.surfaces[ACTIVATION]]).mean(axis=0)
        self.viewport._add("link_line", pv.Line(a, b), color=LINK_COLOR, line_width=4, pickable=False)
        self.viewport._add("link_ends", pv.PolyData(np.array([a, b])), color=LINK_COLOR, point_size=12,
                           render_points_as_spheres=True, pickable=False)

    def open_analysis(self):
        """Show the stored characterisation in the Analysis tab (it reads the saved file)."""
        if not self.project.characterisation:
            QMessageBox.information(self, TITLE, "Run a characterisation first (Record & run tab).")
            return
        if not self.project.path:
            self.save_project(True)
            if not self.project.path:
                return
        else:
            self.save_project(False)
        self.analyse.emit(self.project.path)

    # -----------------------------
    # Out of date components
    # -----------------------------

    def _source_name(self, key):
        path = self.project.neuron_source if key == NEURON else self.project.design_path
        return Path(path).name if path else GROUP_NAMES[key]

    def check_sources(self):
        """Look for saved changes to the components' files (timer)."""
        if self.project.neuron is None and self.project.design_data is None:
            return
        statuses = {k: self.project.reference_status(k) for k in (NEURON, ACTIVATION)}
        if statuses == self.statuses:
            return
        old, self.statuses = self.statuses, statuses
        self._announce_statuses(old)
        self._populate_tree()
        if self.selection and self.selection[0] == "group":
            self._show_properties()
        self._update_actions()

    def _announce_statuses(self, old=None):
        for key, status in self.statuses.items():
            if old is not None and old.get(key) == status:
                continue
            name = self._source_name(key)
            if status == OUTDATED:
                self.log_message(f"{name} was saved with changes: the full neuron still uses the version it imported. "
                                 "Update (toolbar, or right-click its group) to take the new version.")
            elif status == MISSING:
                self.log_message(f"{name} was not found: the copy kept in the full neuron is used.")
            elif status == CURRENT and old is not None and old.get(key) in (OUTDATED, MISSING):
                self.log_message(f"{name} is up to date again.")

    def _add_source_box(self, layout, key):
        status = self.statuses.get(key)
        source = self.project.neuron_source if key == NEURON else self.project.design_path
        if not source:
            return
        box = QGroupBox("Source file")
        v = QVBoxLayout(box)
        label = QLabel(f"{source}<br><b style='color:{STATUS_COLORS.get(status, '#2e7d32')}'>"
                       + {CURRENT: "Up to date", OUTDATED: "Out of date: the file was saved with changes",
                          MISSING: "Not found: the copy kept here is used"}.get(status, "") + "</b>")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(label)
        row = QHBoxLayout()
        update = QPushButton("Update")
        update.setEnabled(status == OUTDATED and self.worker is None)
        update.clicked.connect(lambda _=False, k=key: self.update_component(k))
        row.addWidget(update)
        edit = QPushButton("Open in its tab")
        edit.setEnabled(status != MISSING)
        edit.clicked.connect(lambda _=False, p=source: self.open_source.emit(p))
        row.addWidget(edit)
        row.addStretch(1)
        v.addLayout(row)
        layout.addWidget(box)

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
        status = self.statuses.get(key)
        update = menu.addAction(f"Update from {Path(source).name}")
        update.setEnabled(status == OUTDATED and self.worker is None)
        update.triggered.connect(lambda: self.update_component(key))
        edit = menu.addAction(f"Open {Path(source).name} in its tab")
        edit.setEnabled(status != MISSING)
        edit.triggered.connect(lambda: self.open_source.emit(source))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def update_all(self):
        for key in list(self.project.outdated()):
            if not self.update_component(key):
                break

    def update_component(self, key):
        """Take the saved version of a component's file (True if updated)."""
        if self.worker is not None:
            return False
        name = self._source_name(key)
        try:
            if key == NEURON:
                dialog = UpdateDialog(name, self.project.neuron_changes(), self)
                if dialog.exec_() != QDialog.Accepted:
                    return False
                notes = self.project.update_neuron(keep_conflicts=dialog.keep.isChecked())
                self._load_neuron_cad()
            else:
                notes = self.project.update_design()
                self._load_design_cad()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, TITLE, f"Could not update from {name}:\n{exc}")
            return False
        self.log_message(f"Updated from {name}" + (": " + "; ".join(notes) if notes else "."))
        had_results = bool(self.rows)
        self._after_import()
        if self.project.characterisation:
            self.show_dataset()
            if not self.project.dataset_current():
                self.log_message("The stored characterisation was made with the old version: run it again.")
        elif had_results:
            self.log_message("Solve again for results with the new version.")
        return True

    def closeEvent(self, event):
        self.watch.stop()
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(30000)
        self.viewport.close()
        super().closeEvent(event)


def _center(surfaces):
    pts = np.vstack([m.vertices for m in surfaces.values()])
    return 0.5 * (pts.min(axis=0) + pts.max(axis=0))


def _thinned(frame):
    """Per-vertex thickness of a deformed incompressible rubber sheet: t = t0 A0 / A (nodal areas)."""
    F, rest, x = frame["faces"], frame["rest"], frame["x"]

    def nodal(p):
        area = 0.5 * np.linalg.norm(np.cross(p[F[:, 1]] - p[F[:, 0]], p[F[:, 2]] - p[F[:, 0]]), axis=1)
        out = np.zeros(len(p))
        for k in range(3):
            np.add.at(out, F[:, k], area / 3.0)
        return out
    return float(frame["thickness"]) * nodal(rest) / np.maximum(nodal(x), 1e-300)


def _frame_mesh(frame, scale, place):
    """A body {"kind", "faces", "rest", "x", "thickness"} drawn at its (scaled) deformed shape, placed, with the
    displacement magnitude as "|u| [mm]"."""
    rest, x = frame["rest"], frame["x"]
    u = np.linalg.norm(x - rest, axis=1)
    shown = place(rest + scale * (x - rest))
    if frame["kind"] == "shell":
        t = float(frame.get("thickness") or 0.0)
        if t and frame.get("material") == "neo_hookean":
            t = _thinned(frame)
        pd = solid_shell(shown, frame["faces"], t, u)
    else:
        pd = polydata(shown, frame["faces"])
        pd.point_data["values"] = u
    pd.rename_array("values", "|u| [mm]")
    return pd
