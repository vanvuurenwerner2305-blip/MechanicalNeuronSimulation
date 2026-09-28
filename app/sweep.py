"""Parameter sweep over one or two input chamber pressures (the neuron's response surface)."""
import csv
import time

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                            QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
                            QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .builder import KPA, build_environment, generate_mesh
from .project import CHAMBER, CONSTANT
from .workers import Worker


def sweep_points(a_values, b_values):
    """Grid points in serpentine order, so every solve warm-starts from a close neighbour."""
    points = []
    for i, a in enumerate(a_values):
        js = range(len(b_values)) if i % 2 == 0 else reversed(range(len(b_values)))
        points += [(i, j, a, b_values[j]) for j in js]
    return points


def run_sweep(worker, cad, project, mesh_data, a_index, a_values, b_index, b_values):
    if mesh_data is None:
        worker.log("Generating mesh…")
        mesh_data = generate_mesh(cad, project)
    build = build_environment(cad, mesh_data, project)
    chambers = list(build.volumes.items())
    # one contact stiffness for the whole sweep, sized for its highest pressure
    p_max = max([abs(v.P0) for v in build.volumes.values()] + [abs(x) * KPA for x in a_values]
                + ([abs(x) * KPA for x in b_values] if b_index is not None else []))
    build.set_contact_stiffness(p_max)
    points = sweep_points(list(a_values), list(b_values) if b_index is not None else [None])
    first = True
    for k, (i, j, a, b) in enumerate(points):
        worker.check()
        build.volumes[a_index].P0 = a * KPA
        if b_index is not None:
            build.volumes[b_index].P0 = b * KPA
        callback = lambda lam, it, r: worker.check()
        start = time.time()
        result = None if first else build.solve(project.solver, callback, warm_start=True, load_steps=2,
                                                 fixed_contact=True)
        if result is None or not result.converged:
            result = build.solve(project.solver, callback, fixed_contact=True)
        first = False
        row = {"i": i, "j": j, "a": a, "b": b, "converged": result.converged, "time": time.time() - start,
               "P": {c: v.P / KPA for c, v in chambers}, "dV": {c: v.delta_volume for c, v in chambers}}
        worker.item.emit(row)
        worker.report((k + 1) / len(points), f"Sweep point {k + 1}/{len(points)}")
    return mesh_data


class SweepDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Parameter sweep")
        self.resize(1100, 700)
        self.worker = None
        self.rows = []

        parts = main.project.parts
        self.chambers = [i for i, p in enumerate(parts) if p.role == CHAMBER]
        self.inputs = [i for i in self.chambers if parts[i].props.get("model") == CONSTANT]

        layout = QVBoxLayout(self)
        setup = QGroupBox("Inputs (constant-pressure chambers)")
        grid = QGridLayout(setup)
        grid.addWidget(QLabel("Chamber"), 0, 1)
        grid.addWidget(QLabel("From [kPa]"), 0, 2)
        grid.addWidget(QLabel("To [kPa]"), 0, 3)
        grid.addWidget(QLabel("Points"), 0, 4)
        self.rows_ui = []
        for r, label in enumerate(("Input A", "Input B")):
            enabled = QCheckBox(label)
            enabled.setChecked(r == 0 or len(self.inputs) > 1)
            enabled.setEnabled(r == 1)
            combo = QComboBox()
            for i in self.inputs:
                combo.addItem(parts[i].name, i)
            combo.setCurrentIndex(min(r, len(self.inputs) - 1))
            lo, hi = QDoubleSpinBox(), QDoubleSpinBox()
            for w, v in ((lo, 0.0), (hi, 20.0)):
                w.setRange(-1e6, 1e6)
                w.setDecimals(3)
                w.setValue(v)
            n = QSpinBox()
            n.setRange(1, 200)
            n.setValue(5)
            for c, w in enumerate((enabled, combo, lo, hi, n)):
                grid.addWidget(w, r + 1, c)
            self.rows_ui.append((enabled, combo, lo, hi, n))
        grid.addWidget(QLabel("Plot output"), 3, 0)
        self.output = QComboBox()
        for i in self.chambers:
            self.output.addItem(parts[i].name, i)
        closed = [k for k, i in enumerate(self.chambers) if i not in self.inputs]
        if closed:
            self.output.setCurrentIndex(closed[0])
        self.output.currentIndexChanged.connect(lambda _: self._plot())
        grid.addWidget(self.output, 3, 1)
        layout.addWidget(setup)

        splitter = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        splitter.addWidget(self.table)
        plot = QWidget()
        pl = QVBoxLayout(plot)
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        pl.addWidget(NavigationToolbar2QT(self.canvas, plot))
        pl.addWidget(self.canvas, 1)
        splitter.addWidget(plot)
        splitter.setSizes([450, 650])
        layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.status = QLabel("Each point after the first starts from its neighbour's solution.")
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self.export)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for b in (self.run_button, self.cancel_button, self.export_button, close):
            bottom.addWidget(b)
        layout.addLayout(bottom)

        if not self.inputs:
            self.run_button.setEnabled(False)
            self.status.setText("Set at least one fluid chamber to 'Constant pressure (input)' to sweep it.")

    # -----------------------------

    def _values(self, r):
        _, _, lo, hi, n = self.rows_ui[r]
        return np.linspace(lo.value(), hi.value(), n.value())

    def _setup(self):
        enabled_b, combo_b = self.rows_ui[1][0], self.rows_ui[1][1]
        a_index = self.rows_ui[0][1].currentData()
        b_index = combo_b.currentData() if enabled_b.isChecked() and len(self.inputs) > 1 else None
        if b_index == a_index:
            raise ValueError("Input A and B must be different chambers.")
        return a_index, self._values(0), b_index, self._values(1) if b_index is not None else np.array([np.nan])

    def run(self):
        try:
            self.a_index, self.a_values, self.b_index, self.b_values = self._setup()
        except ValueError as exc:
            QMessageBox.warning(self, "Sweep", str(exc))
            return
        self.rows = []
        parts = self.main.project.parts
        headers = [f"{parts[self.a_index].name} [kPa]"]
        if self.b_index is not None:
            headers.append(f"{parts[self.b_index].name} [kPa]")
        headers += [f"P {parts[c].name} [kPa]" for c in self.chambers] + ["Converged", "Time [s]"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setRowCount(0)

        self.worker = Worker(run_sweep, self.main.cad, self.main.project, self.main.mesh_data_if_current(),
                             self.a_index, self.a_values, self.b_index, self.b_values)
        self.worker.item.connect(self._add_row)
        self.worker.progress.connect(lambda f, t: (self.progress.setValue(int(100 * f)), self.status.setText(t)))
        self.worker.message.connect(self.status.setText)
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()

    def _add_row(self, row):
        self.rows.append(row)
        values = [row["a"]] + ([row["b"]] if self.b_index is not None else [])
        values += [row["P"][c] for c in self.chambers]
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate(values):
            item = QTableWidgetItem(f"{v:.5g}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, c, item)
        self.table.setItem(r, len(values), QTableWidgetItem("yes" if row["converged"] else "NO"))
        self.table.setItem(r, len(values) + 1, QTableWidgetItem(f"{row['time']:.1f}"))
        self.table.scrollToBottom()
        self._plot()

    def _plot(self):
        self.figure.clear()
        if not self.rows:
            self.canvas.draw_idle()
            return
        out = self.output.currentData()
        parts = self.main.project.parts
        ax = self.figure.add_subplot(111)
        if self.b_index is None:
            rows = sorted(self.rows, key=lambda r: r["a"])
            ax.plot([r["a"] for r in rows], [r["P"][out] for r in rows], "o-", color="#2a6fdb")
            ax.set_xlabel(f"{parts[self.a_index].name} [kPa]")
            ax.set_ylabel(f"{parts[out].name} pressure [kPa]")
            ax.grid(alpha=0.3)
        else:
            grid = np.full((len(self.b_values), len(self.a_values)), np.nan)
            for r in self.rows:
                grid[r["j"], r["i"]] = r["P"][out]
            extent = [self.a_values[0], self.a_values[-1], self.b_values[0], self.b_values[-1]]
            if len(self.a_values) == 1:
                extent[:2] = [extent[0] - 0.5, extent[0] + 0.5]
            if len(self.b_values) == 1:
                extent[2:] = [extent[2] - 0.5, extent[2] + 0.5]
            image = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
            self.figure.colorbar(image, ax=ax, label=f"{parts[out].name} pressure [kPa]")
            ax.set_xlabel(f"{parts[self.a_index].name} [kPa]")
            ax.set_ylabel(f"{parts[self.b_index].name} [kPa]")
        self.canvas.draw_idle()

    def _done(self, mesh_data):
        self.main.adopt_mesh(mesh_data)
        self._finish(f"Sweep finished: {len(self.rows)} points, {sum(r['time'] for r in self.rows):.0f} s.")

    def _failed(self, text):
        self._finish("Sweep stopped: " + text.splitlines()[0])
        if text != "Cancelled.":
            self.main.log_message(text)

    def _finish(self, text):
        self.status.setText(text)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    def export(self):
        if not self.rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export sweep", "sweep.csv", "CSV (*.csv)")
        if not path:
            return
        parts = self.main.project.parts
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"{parts[self.a_index].name} [kPa]"]
            if self.b_index is not None:
                header.append(f"{parts[self.b_index].name} [kPa]")
            header += [f"P {parts[c].name} [kPa]" for c in self.chambers]
            header += [f"dV {parts[c].name} [mm3]" for c in self.chambers] + ["converged"]
            writer.writerow(header)
            for r in sorted(self.rows, key=lambda r: (r["a"], r["b"] if r["b"] is not None else 0)):
                writer.writerow([r["a"]] + ([r["b"]] if self.b_index is not None else [])
                                + [r["P"][c] for c in self.chambers] + [r["dV"][c] for c in self.chambers]
                                + [r["converged"]])

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(30000)
        super().closeEvent(event)
