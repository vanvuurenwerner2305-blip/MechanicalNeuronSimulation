"""Main window: import STEP, assign roles, mesh, solve, post-process, sweep."""
import csv
from pathlib import Path

import numpy as np
import pyvista as pv
from qtpy.QtCore import QSettings, Qt, QTimer
from qtpy.QtGui import QKeySequence
try:
    from qtpy.QtGui import QAction, QActionGroup          # Qt 6
except ImportError:
    from qtpy.QtWidgets import QAction, QActionGroup      # Qt 5
from qtpy.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                            QPlainTextEdit, QProgressBar, QPushButton, QSlider, QStyle, QTabWidget, QToolBar)

from .builder import KPA, build_environment, generate_mesh, measure_thickness, mesh_sizes
from .cad import CadModel
from .panels import ModelTree, PropertyPanel, ResultsPanel, SolverPanel
from .project import CHAMBER, DEFORMABLE, ROLE_FIELDS, Project
from .sweep import SweepDialog
from .viewport import Viewport, polydata
from .workers import Worker

APP_NAME = "Membrane Neuron Simulator"
MODEL, MESH, RESULTS = "Model", "Mesh", "Results"

GUIDE = """
<h3>Workflow</h3>
<ol>
<li><b>Model in CAD</b> with every region as its own solid body: each membrane or shell as a thin
solid, each fluid chamber as a solid filling the fluid space, frames/obstacles as solids.
Export the assembly (or multi-body part) as <b>STEP</b>, in millimetres.</li>
<li><b>File → Open STEP</b>. Every solid appears in the model tree.</li>
<li><b>Click a part</b> in the view (Ctrl+click for several) or in the tree and choose its
<b>role</b>: Membrane, Shell, Rigid body, Fluid chamber or Ignore. Parts are see-through (adjust with
the <b>Transparency</b> slider); <b>click the same spot again</b> to select the next part behind.
You can also hide parts with the tree check boxes, or use a <b>section cut</b>.
<i>Edit → Auto-assign roles from names</i> guesses roles from names like "Membrane_1", "Chamber_A".</li>
<li>Set properties: material and thickness of membranes/shells, pressure model and pressure of
chambers. All pressures are gauge: the surroundings are 0 kPa, and a membrane face that touches no
chamber sees 0 kPa.
<ul>
<li><span style="color:#2ca02c"><b>Constant pressure</b></span>: the inputs.</li>
<li><span style="color:#1f6fb4"><b>Ideal gas</b></span>: sealed air, optionally partly filled with
incompressible liquid (darker blue = more liquid).</li>
<li><span style="color:#7b2cbf"><b>Incompressible</b></span>: sealed liquid (stiffness in kPa per %
volume change; less fluid volume than the chamber gives suction, more inflates it).</li>
<li><b>Vent</b> (transparent): open to the surroundings, always 0 kPa.</li>
</ul></li>
<li><b>Mesh</b> (Ctrl+M) shows the simulation mesh: membranes/shells become mid-surfaces, fixed
nodes are blue. <b>Check model</b> reports which chamber acts on which membrane.</li>
<li><b>Solve</b> (F5), then inspect the <b>Results</b> tab. <b>Sweep</b> maps the response over
one or two input pressures.</li>
</ol>
<h3>How parts are simulated</h3>
<ul>
<li>Membranes/shells are clamped along the edges of their mid-surface (or only the edges that touch
rigid bodies). Their largest CAD face defines the mid-surface.</li>
<li>A chamber acts on every membrane/shell whose face it touches.</li>
<li>Rigid bodies are contact obstacles for membranes and shells.</li>
</ul>
<p>Units: mm, N, MPa; pressures in kPa.</p>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)

        self.cad = CadModel()
        self.project = Project()
        self.surfaces = {}           # body index -> SurfaceMesh (display)
        self.mesh_data = None        # simulation mesh
        self.mesh_stale = True
        self.check = None            # BuildResult from "Check model" (couplings, fixed nodes)
        self.build = None            # BuildResult of the last solve
        self.result = None
        self.results_outdated = False
        self.selection = []
        self.mode = MODEL
        self.worker = None
        self._clim = None

        self._build_ui()
        self._build_actions()
        self._update_actions()
        self.log_message("Open a STEP file to start (File → Open STEP…). Help → Quick guide explains the workflow.")

    # -----------------------------
    # UI
    # -----------------------------

    def _build_ui(self):
        self.viewport = Viewport(self)
        self.viewport.picked.connect(self._on_pick)
        self.setCentralWidget(self.viewport)

        self.tree = ModelTree()
        self.tree.selection_changed.connect(lambda idx: self.set_selection(idx, from_tree=True))
        self.tree.visibility_changed.connect(self._on_visibility)
        self.tree.role_requested.connect(self._on_role_changed)
        self.tree.show_only_requested.connect(self._show_only)
        self.tree.show_all_requested.connect(self._show_all)
        dock = QDockWidget("Model", self)
        dock.setObjectName("model_dock")
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.properties = PropertyPanel()
        self.properties.role_changed.connect(self._on_role_changed)
        self.properties.props_changed.connect(self._on_props_changed)
        self.solver_panel = SolverPanel()
        self.solver_panel.changed.connect(self._on_solver_changed)
        self.results_panel = ResultsPanel()
        self.results_panel.display_changed.connect(self._on_results_display)
        self.results_panel.export_vtk.connect(self.export_vtk)
        self.results_panel.export_csv.connect(self.export_csv)
        self.results_panel.screenshot.connect(self.save_screenshot)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.properties, "Part")
        self.tabs.addTab(self.solver_panel, "Solver")
        self.tabs.addTab(self.results_panel, "Results")
        dock = QDockWidget("Properties", self)
        dock.setObjectName("properties_dock")
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        dock = QDockWidget("Messages", self)
        dock.setObjectName("log_dock")
        dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [150], Qt.Vertical)

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

    def _action(self, text, slot, shortcut=None, icon=None, tip=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        action.triggered.connect(slot)
        return action

    def _build_actions(self):
        S = QStyle
        self.a_open_step = self._action("Open STEP…", self.open_step, "Ctrl+O", S.SP_DialogOpenButton,
                                        "Import a STEP assembly or multi-body part")
        self.a_open_project = self._action("Open project…", self.open_project, "Ctrl+Shift+O")
        self.a_save = self._action("Save project", lambda: self.save_project(False), "Ctrl+S",
                                   S.SP_DialogSaveButton, "Save roles, properties and settings")
        self.a_save_as = self._action("Save project as…", lambda: self.save_project(True), "Ctrl+Shift+S")
        self.a_quit = self._action("Exit", self.close, "Ctrl+Q")
        self.a_auto = self._action("Auto-assign roles from names", self.auto_assign, None, None,
                                   "Guess roles for unassigned parts from their names")
        self.a_mesh = self._action("Mesh", self.generate_mesh, "Ctrl+M", S.SP_FileDialogDetailedView,
                                   "Generate the simulation mesh")
        self.a_check = self._action("Check model", self.check_model, "Ctrl+K", S.SP_DialogApplyButton,
                                    "Mesh and report chamber/membrane couplings and warnings")
        self.a_solve = self._action("Solve", self.solve, "F5", S.SP_MediaPlay, "Solve for static equilibrium")
        self.a_sweep = self._action("Sweep…", self.open_sweep, "F6", S.SP_BrowserReload,
                                    "Sweep input chamber pressures")
        self.a_reset_cam = self._action("Reset camera", self.viewport.reset_camera, "R")
        self.a_edges = self._action("Show element edges", self._toggle_edges, "E")
        self.a_edges.setCheckable(True)
        self.a_screenshot = self._action("Screenshot…", self.save_screenshot)
        self.a_guide = self._action("Quick guide", self.show_guide, "F1")
        self.a_about = self._action("About", self.show_about)

        self.view_group = QActionGroup(self)
        self.view_actions = {}
        for mode, key in ((MODEL, "1"), (MESH, "2"), (RESULTS, "3")):
            action = QAction(mode, self, checkable=True)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(lambda _=False, m=mode: self.set_mode(m))
            self.view_group.addAction(action)
            self.view_actions[mode] = action
        self.view_actions[MODEL].setChecked(True)

        menu = self.menuBar()
        m = menu.addMenu("&File")
        for a in (self.a_open_step, self.a_open_project, None, self.a_save, self.a_save_as, None,
                  self.a_screenshot, None, self.a_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = menu.addMenu("&Edit")
        m.addAction(self.a_auto)
        m = menu.addMenu("&View")
        for a in self.view_actions.values():
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_edges)
        m.addAction(self.a_reset_cam)
        m = menu.addMenu("&Simulation")
        for a in (self.a_mesh, self.a_check, self.a_solve, self.a_sweep):
            m.addAction(a)
        m = menu.addMenu("&Help")
        m.addAction(self.a_guide)
        m.addAction(self.a_about)

        tb = QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_open_step, self.a_save, None, self.a_mesh, self.a_check, self.a_solve, self.a_sweep):
            tb.addSeparator() if a is None else tb.addAction(a)
        self.addToolBar(tb)

        vb = QToolBar("View")
        vb.setObjectName("view_toolbar")
        vb.addWidget(QLabel(" View: "))
        for a in self.view_actions.values():
            vb.addAction(a)
        vb.addSeparator()
        vb.addAction(self.a_edges)
        vb.addSeparator()
        vb.addWidget(QLabel(" Transparency: "))
        self.transparency = QSlider(Qt.Horizontal)
        self.transparency.setRange(0, 95)
        self.transparency.setValue(55)
        self.transparency.setFixedWidth(120)
        self.transparency.setToolTip("Transparency of parts that are not selected. "
                                     "Click the same spot again to select the part behind.")
        self.transparency.valueChanged.connect(self._on_transparency)
        vb.addWidget(self.transparency)
        vb.addSeparator()
        vb.addWidget(QLabel(" Section: "))
        self.section_axis = QComboBox()
        self.section_axis.addItems(["Off", "X", "Y", "Z"])
        self.section_axis.currentTextChanged.connect(self._on_section)
        vb.addWidget(self.section_axis)
        self.section_slider = QSlider(Qt.Horizontal)
        self.section_slider.setRange(0, 100)
        self.section_slider.setValue(50)
        self.section_slider.setFixedWidth(160)
        self.section_slider.setEnabled(False)
        self.section_slider.valueChanged.connect(lambda _: self._on_section(self.section_axis.currentText()))
        vb.addWidget(self.section_slider)
        self.addToolBar(vb)

    def _update_actions(self):
        loaded = bool(self.cad.bodies)
        busy = self.worker is not None
        for a in (self.a_save, self.a_save_as, self.a_auto, self.a_mesh, self.a_check, self.a_solve, self.a_sweep):
            a.setEnabled(loaded and not busy)
        for a in (self.a_open_step, self.a_open_project):
            a.setEnabled(not busy)
        self.view_actions[MESH].setEnabled(loaded)
        self.view_actions[RESULTS].setEnabled(self.build is not None)
        name = Path(self.project.path or self.cad.path or "").name
        self.setWindowTitle(f"{APP_NAME} — {name}" if name else APP_NAME)
        mesh = "mesh up to date" if (self.mesh_data is not None and not self.mesh_stale) else "not meshed"
        if loaded:
            assigned = sum(p.role != "Unassigned" for p in self.project.parts)
            self.status_label.setText(f"{len(self.cad.bodies)} parts, {assigned} assigned · {mesh}"
                                      + (" · results outdated" if self.build and self.results_outdated else ""))

    def log_message(self, text):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # -----------------------------
    # Files
    # -----------------------------

    def _settings(self):
        return QSettings("MembraneNeuronSimulator", "App")

    def _dialog_dir(self):
        return self._settings().value("last_dir", str(Path.cwd()))

    def _remember_dir(self, path):
        self._settings().setValue("last_dir", str(Path(path).parent))

    def open_step(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open STEP", self._dialog_dir(), "STEP (*.step *.stp *.STEP *.STP)")
            if not path:
                return
        self._remember_dir(path)
        self._load(path, None)

    def open_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", self._dialog_dir(), "Project (*.mns)")
            if not path:
                return
        self._remember_dir(path)
        try:
            project = Project.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open project", f"Could not read the project:\n{exc}")
            return
        if not Path(project.step_path).exists():
            QMessageBox.critical(self, "Open project", f"STEP file not found:\n{project.step_path}")
            return
        self._load(project.step_path, project)

    def _load(self, step_path, project):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bodies = self.cad.load_step(step_path)
            if project is None:
                project = Project(step_path, [b.name for b in bodies])
            else:
                project.match_bodies(bodies)
            self.project = project
            self.surfaces = self.cad.mesh(mesh_sizes(self.cad, self.project))
            self._detect_thickness(range(len(self.project.parts)))
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Import", f"Could not import the STEP file:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        self.mesh_data, self.mesh_stale, self.check, self.build, self.result = None, True, None, None, None
        self.selection = []
        self.viewport.set_bounds(self.surfaces)
        self.viewport.new_model()
        self.tree.populate(self.project.parts)
        self.solver_panel.load(self.project.solver)
        self.results_panel.set_results([], "No results yet. Press Solve (F5).")
        self.results_panel.set_table([])
        self.log_message(f"Imported {Path(step_path).name}: {len(bodies)} solid bodies "
                         f"({', '.join(b.name for b in bodies)}).")
        if all(p.role == "Unassigned" for p in self.project.parts):
            self.log_message("Click a part to assign its role, or use Edit → Auto-assign roles from names.")
        self.set_mode(MODEL)
        self.set_selection([])
        self._update_actions()

    def save_project(self, save_as=False):
        path = self.project.path
        if save_as or not path:
            default = str(Path(self.cad.path).with_suffix(".mns")) if self.cad.path else "project.mns"
            path, _ = QFileDialog.getSaveFileName(self, "Save project", default, "Project (*.mns)")
            if not path:
                return
        self.project.step_path = self.cad.path
        self.project.save(path)
        self.log_message(f"Saved project {path}")
        self._update_actions()

    # -----------------------------
    # Selection and editing
    # -----------------------------

    def set_selection(self, indices, from_tree=False):
        self.selection = sorted(set(indices))
        if not from_tree:
            self.tree.select(self.selection)
        self.properties.set_selection(self.selection, self.project.parts, self.cad.bodies,
                                      self._couplings_text(self.selection))
        if self.selection:
            self.tabs.setCurrentWidget(self.properties)
        self.refresh_view()

    def _on_pick(self, index, add):
        if index < 0:  # clicked on empty space
            self.set_selection([])
            return
        if add:
            indices = set(self.selection) ^ {index}
        else:
            indices = {index}
        self.set_selection(sorted(indices))

    def _on_visibility(self, index, visible):
        self.project.parts[index].visible = visible
        self.refresh_view()

    def _show_only(self, indices):
        for i, part in enumerate(self.project.parts):
            part.visible = i in indices
        self.tree.refresh(self.project.parts)
        self.refresh_view()

    def _show_all(self):
        for part in self.project.parts:
            part.visible = True
        self.tree.refresh(self.project.parts)
        self.refresh_view()

    def _invalidate(self, mesh):
        if mesh:
            self.mesh_stale = True
        self.check = None
        if self.build is not None:
            self.results_outdated = True
        self._update_actions()

    def _detect_thickness(self, indices):
        """Give membranes/shells without a thickness the one measured from their CAD solid, and
        chambers without a fluid volume the body's volume."""
        for i in indices:
            part = self.project.parts[i]
            if part.role == CHAMBER and not float(part.props.get("fluid_volume", 0.0) or 0.0):
                part.props["fluid_volume"] = round(self.cad.bodies[i].volume, 6)
            if part.role in DEFORMABLE and not float(part.props.get("thickness", 0.0) or 0.0):
                try:
                    part.props["thickness"] = round(measure_thickness(self.cad, self.surfaces, i), 6)
                    self.log_message(f"{part.name}: measured thickness {part.props['thickness']:.4g} mm")
                except Exception as exc:  # leave 0: measured again during meshing
                    self.log_message(f"{part.name}: could not measure the thickness ({exc})")

    def _on_role_changed(self, indices, role):
        for i in indices:
            self.project.parts[i].set_role(role)
        self._detect_thickness(indices)
        self.log_message(f"{', '.join(self.project.parts[i].name for i in indices)} → {role}")
        self.tree.refresh(self.project.parts)
        self._invalidate(mesh=True)
        self.set_selection(indices)

    def _on_props_changed(self, indices, key, value):
        mesh_field = False
        rebuild = False
        for i in indices:
            part = self.project.parts[i]
            spec = {f.key: f for f in ROLE_FIELDS[part.role]}.get(key)
            if spec is None or part.props.get(key) == value:
                continue
            part.props[key] = value
            mesh_field |= spec.mesh
            rebuild |= spec.kind == "choice"
        self._invalidate(mesh=mesh_field)
        self.tree.refresh(self.project.parts)  # colours follow chamber models
        QTimer.singleShot(0, self.refresh_view)
        if rebuild:  # choices can show or hide other fields; rebuild once this event is done
            QTimer.singleShot(0, lambda: self.properties.set_selection(
                self.selection, self.project.parts, self.cad.bodies, self._couplings_text(self.selection)))

    def _on_solver_changed(self, key, value):
        setattr(self.project.solver, key, value)
        self._invalidate(mesh=False)

    def auto_assign(self):
        changed = self.project.auto_assign_from_names()
        self._detect_thickness(range(len(self.project.parts)))
        self.log_message(f"Auto-assigned {changed} part(s) from their names. Check the roles in the model tree.")
        self.tree.refresh(self.project.parts)
        self._invalidate(mesh=True)
        self.set_selection(self.selection)

    def _couplings_text(self, indices):
        build = self.check or self.build
        if len(indices) != 1 or build is None or self.project.parts[indices[0]].role != CHAMBER:
            return None if len(indices) != 1 else "Run Check model (Ctrl+K) to see which membranes this chamber loads."
        couplings = build.couplings.get(indices[0], [])
        if not couplings:
            return "No membrane or shell touches this chamber."
        return "\n".join(f"• {self.project.parts[c.shell_index].name} ({c.coverage:.0%} of its face)"
                         for c in couplings)

    # -----------------------------
    # Views
    # -----------------------------

    def set_mode(self, mode):
        if mode == RESULTS and self.build is None:
            mode = MODEL
        self.mode = mode
        self.view_actions[mode].setChecked(True)
        if mode == MESH and (self.mesh_data is None or self.mesh_stale) and self.cad.bodies and self.worker is None:
            self.generate_mesh()
        self.refresh_view()

    def refresh_view(self):
        if not self.surfaces:
            return
        parts = self.project.parts
        if self.mode == RESULTS and self.build is not None:
            step = self.results_panel.current_step()
            if step is None:
                return
            field = self.results_panel.field.currentText()
            clim = self._global_clim(field) if self.results_panel.fixed_range.isChecked() else None
            self.viewport.show_results(self.mesh_data.surfaces, parts, self.build, step, field,
                                       self.results_panel.scale.value(), clim)
            self._fill_results_table(step)
        elif self.mode == MESH and self.mesh_data is not None:
            self.viewport.show_mesh(self.mesh_data.surfaces, parts, self.selection, self.mesh_data, self.check)
        else:
            self.viewport.show_model(self.surfaces, parts, self.selection)

    def _global_clim(self, field):
        key = (id(self.build), field)
        if self._clim is None or self._clim[0] != key:
            values = [v for step in self.build.env.history
                      for v, _ in self.viewport.result_values(self.build, step, field).values()]
            allv = np.concatenate(values)
            lo, hi = float(allv.min()), float(allv.max())
            self._clim = (key, (lo, hi if hi > lo else lo + 1e-12))
        return self._clim[1]

    def _toggle_edges(self):
        self.viewport.show_edges = self.a_edges.isChecked()
        self.refresh_view()

    def _on_transparency(self, value):
        self.viewport.opacity = 1.0 - value / 100.0
        self.refresh_view()

    def _on_section(self, axis):
        self.section_slider.setEnabled(axis != "Off")
        self.viewport.section = None if axis == "Off" else (axis, self.section_slider.value() / 100.0)
        self.refresh_view()

    def _on_results_display(self):
        if self.mode == RESULTS:
            self.refresh_view()

    # -----------------------------
    # Jobs
    # -----------------------------

    def mesh_data_if_current(self):
        return None if self.mesh_stale else self.mesh_data

    def adopt_mesh(self, mesh_data):
        if mesh_data is not None and mesh_data is not self.mesh_data:
            self.mesh_data, self.mesh_stale = mesh_data, False
            self.surfaces = mesh_data.surfaces
            self._update_actions()

    def _start(self, fn, args, on_success, text):
        self.worker = Worker(fn, *args, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self.log_message)
        self.worker.succeeded.connect(lambda result: self._job_done(on_success, result))
        self.worker.failed.connect(self._job_failed)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.status_label.setText(text)
        self.log_message(text)
        self._update_actions()
        self.worker.start()

    def _on_progress(self, fraction, text):
        self.progress.setRange(0, 100 if fraction >= 0 else 0)
        if fraction >= 0:
            self.progress.setValue(int(100 * fraction))
        self.status_label.setText(text)

    def _job_done(self, on_success, result):
        self.worker = None
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        try:
            on_success(result)
        finally:
            self._update_actions()

    def _job_failed(self, text):
        self.worker = None
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.log_message("Stopped: " + text)
        if text != "Cancelled.":
            QMessageBox.warning(self, APP_NAME, text.split("\n\n")[0])
        self._update_actions()

    def generate_mesh(self):
        def job(worker, cad, project):
            worker.report(-1, "Meshing…")
            data = generate_mesh(cad, project)
            return data, build_environment(cad, data, project)
        self._start(job, (self.cad, self.project), self._mesh_done, "Generating mesh…")

    def _mesh_done(self, result):
        data, check = result
        self.adopt_mesh(data)
        self.check = check
        n = sum(len(m.faces) for m in data.midsurfaces.values())
        self.log_message(f"Mesh: {n} shell elements on {len(data.midsurfaces)} membrane/shell part(s).")
        for i, mid in data.midsurfaces.items():
            self.log_message(f"  {self.project.parts[i].name}: {len(mid.faces)} elements, "
                             f"measured thickness {mid.thickness:.4g} mm")
        self.mode = MESH
        self.view_actions[MESH].setChecked(True)
        self.set_selection(self.selection)

    def check_model(self):
        def job(worker, cad, project, mesh):
            worker.report(-1, "Checking model…")
            mesh = mesh or generate_mesh(cad, project)
            return mesh, build_environment(cad, mesh, project)
        self._start(job, (self.cad, self.project, self.mesh_data_if_current()), self._check_done, "Checking model…")

    def _check_done(self, result):
        data, check = result
        self.adopt_mesh(data)
        self.check = check
        parts = self.project.parts
        self.log_message("Model check:")
        for c, couplings in check.couplings.items():
            targets = ", ".join(f"{parts[k.shell_index].name}" for k in couplings) or "nothing"
            self.log_message(f"  {parts[c].name} acts on {targets}")
        for i, t in check.thickness.items():
            fixed = int(check.shells[i].fixed.sum())
            self.log_message(f"  {parts[i].name}: thickness {t:.4g} mm, {fixed} fixed nodes")
        self.log_message(f"  Contact stiffness {check.contact_stiffness:.4g} MPa/mm")
        for w in check.warnings:
            self.log_message("  Warning: " + w)
        if not check.warnings:
            self.log_message("  No problems found.")
        self.mode = MESH
        self.view_actions[MESH].setChecked(True)
        self.set_selection(self.selection)

    def solve(self):
        def job(worker, cad, project, mesh):
            if mesh is None:
                worker.report(-1, "Meshing…")
                mesh = generate_mesh(cad, project)
            build = build_environment(cad, mesh, project)
            for w in build.warnings:
                worker.log("Warning: " + w)

            def callback(lam, iteration, residual):
                worker.check()
                worker.report(lam, f"Solving: load {lam:.0%}, Newton iteration {iteration}, |R| = {residual:.2e}")
            return mesh, build, build.solve(project.solver, callback=callback)
        self._start(job, (self.cad, self.project, self.mesh_data_if_current()), self._solve_done, "Solving…")

    def _solve_done(self, result):
        data, build, solve_result = result
        self.adopt_mesh(data)
        self.build, self.result, self.check = build, solve_result, build
        self.results_outdated = False
        self._clim = None
        parts = self.project.parts
        status = ("Converged" if solve_result.converged else "NOT converged") + \
                 f" · {len(solve_result.load_factors)} load steps · {sum(solve_result.iterations)} Newton iterations · " \
                 f"{solve_result.message}"
        self.log_message(status)
        for c, v in build.volumes.items():
            self.log_message(f"  {parts[c].name}: P = {v.P / KPA:.5g} kPa, ΔV = {v.delta_volume:.5g} mm³")
        if not solve_result.converged:
            QMessageBox.warning(self, APP_NAME, "The solver did not reach full load. Results show the last "
                                                "converged load step.\n\n" + solve_result.message)
        self.results_panel.set_results(build.env.history, status)
        self.tabs.setCurrentWidget(self.results_panel)
        self.set_mode(RESULTS)

    def _fill_results_table(self, step):
        parts = self.project.parts
        rows = []
        for (c, v), P, dV in zip(self.build.volumes.items(), step["pressures"], step["delta_volumes"]):
            rows.append([parts[c].name, P / KPA, dV, (v.initial_volume or 0) + dV])
        disp = []
        for (i, shell), coords in zip(self.build.shells.items(), step["shell_coords"]):
            u = np.linalg.norm(coords.numpy() - shell.X.cpu().numpy(), axis=1).max()
            disp.append(f"{parts[i].name}: max displacement {u:.4g} mm")
        self.results_panel.set_table(rows, "\n".join(disp))

    def open_sweep(self):
        if not any(p.role == CHAMBER for p in self.project.parts):
            QMessageBox.information(self, "Sweep", "Assign fluid chambers first.")
            return
        SweepDialog(self, self).exec()
        self._update_actions()

    # -----------------------------
    # Export
    # -----------------------------

    def export_vtk(self):
        if self.build is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", "results.vtm", "VTK multiblock (*.vtm)")
        if not path:
            return
        step = self.results_panel.current_step()
        blocks = pv.MultiBlock()
        for (i, shell), coords in zip(self.build.shells.items(), step["shell_coords"]):
            X = shell.X.cpu().numpy()
            pd = polydata(coords.numpy(), shell.faces.cpu().numpy())
            pd.point_data["displacement"] = coords.numpy() - X
            stretch, _ = self.viewport.result_values(self.build, step, "Area stretch")[i]
            pd.cell_data["area_stretch"] = stretch
            blocks[self.project.parts[i].name] = pd
        for i, mesh in self.mesh_data.surfaces.items():
            if i in self.build.obstacles:
                blocks[self.project.parts[i].name] = polydata(mesh.vertices, mesh.faces)
        blocks.save(path)
        self.log_message(f"Exported {path}")

    def export_csv(self):
        if self.build is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export chamber results", "chambers.csv", "CSV (*.csv)")
        if not path:
            return
        names = [self.project.parts[c].name for c in self.build.volumes]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["load_factor"] + [f"P {n} [kPa]" for n in names] + [f"dV {n} [mm3]" for n in names])
            for step in self.build.env.history:
                writer.writerow([step["load_factor"]] + [p / KPA for p in step["pressures"]] + step["delta_volumes"])
        self.log_message(f"Exported {path}")

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Screenshot", "view.png", "PNG (*.png)")
        if path:
            self.viewport.screenshot(path)
            self.log_message(f"Saved {path}")

    # -----------------------------
    # Help / lifetime
    # -----------------------------

    def show_guide(self):
        box = QMessageBox(self)
        box.setWindowTitle("Quick guide")
        box.setTextFormat(Qt.RichText)
        box.setText(GUIDE)
        box.exec()

    def show_about(self):
        QMessageBox.about(self, APP_NAME, f"<b>{APP_NAME}</b><br>Static simulation of fluid-driven membranes "
                                          "and shells: large-strain shell elements, fluid chambers, rigid contact, "
                                          "Newton-Raphson solver.")

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(30000)
        self.viewport.close()
        self.cad.close()
        super().closeEvent(event)
