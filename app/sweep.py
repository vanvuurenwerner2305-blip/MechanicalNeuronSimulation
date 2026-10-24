"""The parameter-sweep dialog of the neuron space: one or two input chamber pressures, with the chamber pressures and
any linked activation design's outputs at every point. The computation (run_sweep, the CSV) is in app.sweep_core,
shared with the command-line API.
"""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                            QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                            QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                            QVBoxLayout, QWidget)

from .project import ACTIVATION_MEMBRANE, CHAMBER, CONSTANT, VENT
from .sweep_core import activation_keys, activation_label, activation_value, run_sweep, write_sweep_csv
from .workers import Worker


class SweepDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Parameter sweep")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        self.resize(min(1400, int(0.9 * available.width())) if available else 1300,
                    min(900, int(0.9 * available.height())) if available else 850)
        self.worker = None
        self.rows = []
        self._closing = False

        parts = main.project.parts
        self.chambers = [i for i, p in enumerate(parts) if p.role == CHAMBER]
        self.inputs = [i for i in self.chambers if parts[i].props.get("model") == CONSTANT]
        self.closed = [i for i in self.chambers if parts[i].props.get("model") not in (CONSTANT, VENT)]
        self.linked = [i for i, p in enumerate(parts) if p.role == ACTIVATION_MEMBRANE]

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
        self.output.currentIndexChanged.connect(lambda _: self._plot())
        grid.addWidget(self.output, 3, 1, 1, 3)
        self._fill_outputs()
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
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([480, 900])
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

    def _fill_outputs(self):
        """Plot choices: chamber pressures, then the linked designs' outputs."""
        current = self.output.currentText()
        parts = self.main.project.parts
        self.output.blockSignals(True)
        self.output.clear()
        for i in self.chambers:
            self.output.addItem(f"P {parts[i].name}", ("P", i))
        for i in self.linked:
            for field in activation_keys(self.rows, i):
                name, unit = activation_label(field)
                self.output.addItem(f"{parts[i].name}: {name} [{unit}]", ("act", (i, field)))
        k = self.output.findText(current)
        if k < 0:
            closed = [n for n, i in enumerate(self.chambers) if i in self.closed]
            k = closed[0] if closed else 0
        self.output.setCurrentIndex(k)
        self.output.blockSignals(False)

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
        self._fill_outputs()
        self.table.setColumnCount(0)
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

    def _headers(self):
        parts = self.main.project.parts
        headers = [f"{parts[self.a_index].name} [kPa]"]
        if self.b_index is not None:
            headers.append(f"{parts[self.b_index].name} [kPa]")
        headers += [f"P {parts[c].name} [kPa]" for c in self.chambers]
        headers += [f"{parts[i].name} {'{} [{}]'.format(*activation_label(k))}" for i in self.linked
                    for k in activation_keys(self.rows, i)]
        return headers + ["Converged", "Time [s]"]

    def _add_row(self, row):
        self.rows.append(row)
        if len(self.rows) == 1:
            self._fill_outputs()
            headers = self._headers()
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        values = [row["a"]] + ([row["b"]] if self.b_index is not None else [])
        values += [row["P"][c] for c in self.chambers]
        values += [activation_value(row, i, k) for i in self.linked for k in activation_keys(self.rows, i)]
        r = self.table.rowCount()
        self.table.insertRow(r)
        warning = "\n".join(row.get("extrapolated") or [])
        for c, v in enumerate(values):
            item = QTableWidgetItem(f"{v:.5g}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if warning:  # outside an activation design's simulated range: extrapolated
                item.setBackground(QColor("#ffe0b2"))
                item.setToolTip(warning)
            self.table.setItem(r, c, item)
        self.table.setItem(r, len(values), QTableWidgetItem("yes" if row["converged"] else "NO"))
        self.table.setItem(r, len(values) + 1, QTableWidgetItem(f"{row['time']:.1f}"))
        self.table.scrollToBottom()
        self._plot()

    def _plot(self):
        self.figure.clear()
        data = self.output.currentData()
        if not self.rows or data is None:
            self.canvas.draw_idle()
            return
        field, key = data
        parts = self.main.project.parts
        ax = self.figure.add_subplot(111)
        if field == "P":
            value, label = (lambda r: r["P"][key]), f"{parts[key].name} pressure [kPa]"
        else:
            name, unit = activation_label(key[1])
            value, label = (lambda r: activation_value(r, *key)), f"{parts[key[0]].name}: {name} [{unit}]"
        if self.b_index is None:
            rows = sorted(self.rows, key=lambda r: r["a"])
            ax.plot([r["a"] for r in rows], [value(r) for r in rows], "o-", color="#2a6fdb")
            ax.set_xlabel(f"{parts[self.a_index].name} [kPa]")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        else:
            grid = np.full((len(self.b_values), len(self.a_values)), np.nan)
            for r in self.rows:
                grid[r["j"], r["i"]] = value(r)
            extent = [self.a_values[0], self.a_values[-1], self.b_values[0], self.b_values[-1]]
            if len(self.a_values) == 1:
                extent[:2] = [extent[0] - 0.5, extent[0] + 0.5]
            if len(self.b_values) == 1:
                extent[2:] = [extent[2] - 0.5, extent[2] + 0.5]
            image = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
            self.figure.colorbar(image, ax=ax, label=label)
            ax.set_xlabel(f"{parts[self.a_index].name} [kPa]")
            ax.set_ylabel(f"{parts[self.b_index].name} [kPa]")
        self.canvas.draw_idle()

    def _done(self, mesh_data):
        self.main.adopt_mesh(mesh_data)
        text = f"Sweep finished: {len(self.rows)} points, {sum(r['time'] for r in self.rows):.0f} s."
        outside = sum(bool(r.get("extrapolated")) for r in self.rows)
        if outside:
            text += (f" ⚠ {outside} point(s) outside an activation design's simulated Δp range: EXTRAPOLATING "
                     "(orange rows).")
            QMessageBox.warning(self, "Sweep", f"{outside} of {len(self.rows)} point(s) put the pre-activation Δp "
                                               "outside the activation design's simulated range: there the design "
                                               "is extrapolated, not interpolated (orange rows; hover for details).")
        self._finish(text)

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
        write_sweep_csv(path, self.rows, self.main.project.parts, self.a_index, self.b_index, self.chambers,
                        self.linked)
        self.status.setText(f"Exported {path}.")

    # -----------------------------
    # Closing while a sweep runs: cancel it and close once its thread has finished (waiting for it here would
    # freeze the window; Windows then reports the app as not responding)
    # -----------------------------

    def _busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _stop_and_close(self):
        self._closing = True
        self.worker.cancel()
        self.status.setText("Stopping…")
        self.worker.finished.connect(lambda: QTimer.singleShot(0, self.reject))

    def reject(self):  # Esc, the Close button and the window's close button all end here
        if self._busy():
            self._stop_and_close()
            return
        super().reject()

    def closeEvent(self, event):
        if self._busy():
            event.ignore()
            self._stop_and_close()
            return
        super().closeEvent(event)
