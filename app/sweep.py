"""Parameter sweep over one or two input chamber pressures (the neuron's response surface).

Besides the chamber pressures, every point records the mechanical weight of every input path into
the chosen activation chamber: W_j = dV_j / (p_j - p_a), with dV_j the volume the path pushes into
the activation chamber (see membrane_sim.characterise; a path lumps its membranes and intermediate
chambers), plus the activation chamber's own compliance W_0. After the sweep each W is fitted as
the lowest-degree polynomial W(dp) that passes within the tolerance of every point (degree 0: W is
constant; points with dp = 0 are left out), and the neuron equation is written out in LaTeX.
"""
import csv
import math
import os
import time

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                            QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                            QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
                            QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from membrane_sim.characterise import (BIGGEST_ERROR, LOWEST_TOTAL, fit_neuron_equation, input_weights,
                                       equation_align, equation_lines, neuron_equation_latex, polynomial_text,
                                       rebuild_activation_pressure, weight_degree)

from .builder import KPA, build_environment, generate_mesh
from .project import CHAMBER, CONSTANT, VENT
from .workers import Worker

FLUID = "W0"  # key of the activation chamber's own compliance among the weights
# per-path quantities: key -> (label, unit, factor from model units: mm, MPa)
WEIGHT_FIELDS = {
    "p": ("p", "kPa", 1.0 / KPA),
    "dp": ("dp", "kPa", 1.0 / KPA),
    "dV": ("dV", "mm3", 1.0),
    "W": ("W", "mm3/kPa", KPA),
    "W_tan": ("dV/dp tangent", "mm3/kPa", KPA),
}


def point_weights(build, activation_index, load_factor):
    """Weights of every input path (label -> {key: value in WEIGHT_FIELDS units, "shells": names}),
    the activation chamber's own term under FLUID, and the activation pressure rebuilt from them."""
    raw = input_weights(build.env, build.volumes[activation_index], load_factor)
    scale = lambda w: {k: w[k] * f for k, (_, _, f) in WEIGHT_FIELDS.items()}
    weights = {w["input"]: {**scale(w), "shells": w["shells"]} for w in raw["inputs"]}
    weights[FLUID] = {**scale(raw["chamber"]), "shells": []}
    return weights, rebuild_activation_pressure(raw) / KPA


def equation_fit(rows, activation_index, tolerance, method=LOWEST_TOTAL):
    """Fit the weights of the converged rows so the equation meets `tolerance` (kPa) on p_a."""
    samples = [{"p_a": r["P"][activation_index],
                "terms": {k: (w["p"], w["dp"], w["W"]) for k, w in r["W"].items()}}
               for r in rows if r["converged"] and r["W"]]
    return fit_neuron_equation(samples, tolerance, method)


NO_FIT = {"kind": "none", "sides": {"+": None, "-": None}}
SIDE_NAME = {"+": "dp > 0", "-": "dp < 0"}
SIDE_COLOR = {"+": "#d9480f", "-": "#7048e8"}


def describe_side(fit):
    if fit["degree"] == 0:
        return f"W = {fit['coefficients'][0]:.6g} mm3/kPa, constant ({fit['points']} points)"
    return (f"W(dp) = {polynomial_text(fit['coefficients'], precision=6)}, degree {fit['degree']} "
            f"({fit['points']} points)")


def describe_fit(fit):
    """One text per side of a piecewise weight."""
    sides = [side for side in ("+", "-") if fit["sides"][side] is not None]
    if not sides:
        return "no points with a pressure difference (W = 0)"
    if len(sides) == 1:
        return f"{describe_side(fit['sides'][sides[0]])}, for all dp (sampled only {SIDE_NAME[sides[0]]})"
    return "; ".join(f"{SIDE_NAME[side]}: {describe_side(fit['sides'][side])}" for side in sides)


def sweep_points(a_values, b_values):
    """Grid points in serpentine order, so every solve warm-starts from a close neighbour."""
    points = []
    for i, a in enumerate(a_values):
        js = range(len(b_values)) if i % 2 == 0 else reversed(range(len(b_values)))
        points += [(i, j, a, b_values[j]) for j in js]
    return points


def run_sweep(worker, cad, project, mesh_data, a_index, a_values, b_index, b_values, activation_index=None):
    if mesh_data is None:
        worker.log("Generating mesh…")
        mesh_data = generate_mesh(cad, project)
    build = build_environment(cad, mesh_data, project)
    chambers = list(build.volumes.items())
    if activation_index is not None and not build.volumes[activation_index].is_closed:
        raise ValueError(f"{project.parts[activation_index].name} is not a closed chamber: "
                         "it can not be the activation chamber.")
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
               "P": {c: v.P / KPA for c, v in chambers}, "dV": {c: v.delta_volume for c, v in chambers},
               "W": {}, "p_a_rebuilt": math.nan}
        if activation_index is not None:
            row["W"], row["p_a_rebuilt"] = point_weights(build, activation_index, result.load_factor)
        worker.item.emit(row)
        worker.report((k + 1) / len(points), f"Sweep point {k + 1}/{len(points)}")
    return mesh_data


class SweepDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Parameter sweep")
        self.resize(1150, 760)
        self.worker = None
        self.rows = []
        self.weight_keys = []
        self.activation_index = None
        self._equation = None    # equation_fit of the current rows (set by the fit worker)
        self.fit_worker = None   # background equation fit (the exhaustive search can take seconds)

        parts = main.project.parts
        self.chambers = [i for i, p in enumerate(parts) if p.role == CHAMBER]
        self.inputs = [i for i in self.chambers if parts[i].props.get("model") == CONSTANT]
        self.closed = [i for i in self.chambers if parts[i].props.get("model") not in (CONSTANT, VENT)]

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

        grid.addWidget(QLabel("Activation chamber"), 3, 0)
        self.activation = QComboBox()
        self.activation.addItem("(none: no weights)", None)
        for i in self.closed:
            self.activation.addItem(parts[i].name, i)
        named = [k for k, i in enumerate(self.closed) if "activ" in parts[i].name.lower()]
        self.activation.setCurrentIndex(1 + (named[0] if named else 0) if self.closed else 0)
        self.activation.setToolTip("The chamber whose pressure p_a is the neuron's activation. Every input path "
                                   "into it gets one weight W_j = dV_j / (p_j - p_a).")
        grid.addWidget(self.activation, 3, 1)
        grid.addWidget(QLabel("Equation tolerance [kPa]"), 3, 2)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(1e-4, 1e4)
        self.tolerance.setDecimals(4)
        self.tolerance.setValue(1.0)
        self.tolerance.setToolTip("The weights W(dp) get the lowest polynomial degrees (constant first) for which the "
                                  "neuron equation, solved for p_a, is within this of every simulated p_a.")
        grid.addWidget(self.tolerance, 3, 3)
        self.regenerate = QPushButton("Regenerate equation")
        self.regenerate.setToolTip("Refit the weights to the current tolerance from the sweep's results "
                                   "(no new simulations).")
        self.regenerate.setEnabled(False)
        self.regenerate.clicked.connect(self._refit)
        grid.addWidget(self.regenerate, 3, 4)
        grid.addWidget(QLabel("Degree search"), 5, 0)
        self.method = QComboBox()
        self.method.addItem("Lowest total order (exhaustive)", LOWEST_TOTAL)
        self.method.addItem("Biggest own error first (greedy)", BIGGEST_ERROR)
        self.method.setToolTip("Lowest total order tries every combination of degrees, total order 0, 1, 2, ..., "
                               "and is guaranteed minimal.\nBiggest own error first raises the weight that causes "
                               "the largest error on its own; faster, but can end at a higher total order.")
        grid.addWidget(self.method, 5, 1, 1, 2)
        grid.addWidget(QLabel("Plot output"), 4, 0)
        self.output = QComboBox()
        self.output.currentIndexChanged.connect(lambda _: self._plot())
        grid.addWidget(self.output, 4, 1, 1, 3)
        self._fill_outputs()
        layout.addWidget(setup)

        splitter = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        splitter.addWidget(self.table)
        self.tabs = QTabWidget()
        plot = QWidget()
        pl = QVBoxLayout(plot)
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        pl.addWidget(NavigationToolbar2QT(self.canvas, plot))
        pl.addWidget(self.canvas, 1)
        self.tabs.addTab(plot, "Plot")
        equation = QWidget()
        el = QVBoxLayout(equation)
        parts_split = QSplitter(Qt.Vertical)
        self.eq_figure = Figure(figsize=(6, 3))
        self.eq_canvas = FigureCanvasQTAgg(self.eq_figure)
        self.eq_canvas.setToolTip("Click a weight to plot its fitted polynomial over the sampled points.")
        self.eq_canvas.mpl_connect("pick_event", self._pick_weight)
        parts_split.addWidget(self.eq_canvas)
        self.weight_figure = Figure(figsize=(6, 2.5), tight_layout=True)
        self.weight_canvas = FigureCanvasQTAgg(self.weight_figure)
        parts_split.addWidget(self.weight_canvas)
        self.latex = QPlainTextEdit()
        self.latex.setReadOnly(True)
        self.latex.setPlaceholderText("The neuron equation (LaTeX) appears here after a sweep with an activation chamber.")
        parts_split.addWidget(self.latex)
        parts_split.setSizes([260, 260, 140])
        el.addWidget(parts_split, 1)
        self.selected_weight = None  # key of the weight shown under the equation
        copy = QPushButton("Copy LaTeX")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.latex.toPlainText()))
        el.addWidget(copy, 0, Qt.AlignRight)
        self.tabs.addTab(equation, "Neuron equation")
        splitter.addWidget(self.tabs)
        splitter.setSizes([450, 700])
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
        self.export_button = QPushButton("Export…")
        self.export_button.setToolTip("<name>.csv (all points), <name>_weights.csv (W fits), <name>_equation.tex")
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

    def _weight_label(self, key):
        return f"{self.main.project.parts[self.activation_index].name} fluid (W0)" if key == FLUID else key

    def _fill_outputs(self):
        """Plot choices: chamber pressures, then per weight its fit and raw quantities."""
        current = self.output.currentText()
        parts = self.main.project.parts
        self.output.blockSignals(True)
        self.output.clear()
        for i in self.chambers:
            self.output.addItem(f"P {parts[i].name}", ("P", i))
        if self.weight_keys:
            self.output.addItem("p_a from the fitted equation - simulated", ("equation", None))
            self.output.addItem("p_a from the point's own weights - simulated (check)", ("rebuilt", None))
        for key in self.weight_keys:
            label = self._weight_label(key)
            self.output.addItem(f"W vs dp: {label} (with fit)", ("fit", key))
            for field in ("dp", "dV", "W", "W_tan"):
                name, unit, _ = WEIGHT_FIELDS[field]
                self.output.addItem(f"{name}: {label} [{unit}]", (field, key))
        k = self.output.findText(current)
        if k < 0:
            default = [n for n in range(self.output.count()) if self.output.itemData(n)[0] == "fit"]
            closed = [n for n, i in enumerate(self.chambers) if i in self.closed]
            k = default[0] if default else closed[0] if closed else 0
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
        self.activation_index = self.activation.currentData()
        self.rows, self.weight_keys = [], []
        self._fill_outputs()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.latex.clear()
        self.eq_figure.clear()
        self.eq_canvas.draw_idle()

        self.worker = Worker(run_sweep, self.main.cad, self.main.project, self.main.mesh_data_if_current(),
                             self.a_index, self.a_values, self.b_index, self.b_values, self.activation_index)
        self.worker.item.connect(self._add_row)
        self.worker.progress.connect(lambda f, t: (self.progress.setValue(int(100 * f)), self.status.setText(t)))
        self.worker.message.connect(self.status.setText)
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.run_button.setEnabled(False)
        self.regenerate.setEnabled(False)
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
        if self.weight_keys:
            headers.append("p_a from weights [kPa]")
        headers += [f"W {self._weight_label(k)} [mm3/kPa]" for k in self.weight_keys]
        return headers + ["Converged", "Time [s]"]

    def _add_row(self, row):
        self.rows.append(row)
        self._equation = None
        if len(self.rows) == 1:
            self.weight_keys = [k for k in row["W"] if k != FLUID] + ([FLUID] if FLUID in row["W"] else [])
            self._fill_outputs()
            headers = self._headers()
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        values = [row["a"]] + ([row["b"]] if self.b_index is not None else [])
        values += [row["P"][c] for c in self.chambers]
        if self.weight_keys:
            values.append(row["p_a_rebuilt"])
        values += [row["W"].get(k, {}).get("W", math.nan) for k in self.weight_keys]
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
        data = self.output.currentData()
        if not self.rows or data is None:
            self.canvas.draw_idle()
            return
        field, key = data
        parts = self.main.project.parts
        ax = self.figure.add_subplot(111)
        if field == "fit":
            self._plot_fit(ax, key)
            self.canvas.draw_idle()
            return
        if field == "P":
            value, label = (lambda r: r["P"][key]), f"{parts[key].name} pressure [kPa]"
        elif field == "equation":
            fit, rows = self.equation(), [r for r in self.rows if r["converged"] and r["W"]]
            error = {id(r): e for r, e in zip(rows, fit["predicted"])} if fit and len(fit["predicted"]) == len(rows) else {}
            value = lambda r: error.get(id(r), math.nan) - r["P"][self.activation_index]
            label = "p_a from equation - simulated [kPa]"
        elif field == "rebuilt":
            value, label = (lambda r: r["p_a_rebuilt"] - r["P"][self.activation_index]), \
                "p_a from weights - simulated p_a [kPa]"
        else:
            name, unit = WEIGHT_FIELDS[field][:2]
            value, label = (lambda r: r["W"].get(key, {}).get(field, math.nan)), \
                f"{name}: {self._weight_label(key)} [{unit}]"
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

    def _plot_fit(self, ax, key):
        """A weight against its pressure difference, with the fitted polynomial on top."""
        fit = self.equation()["fits"].get(key, NO_FIT) if self.equation() else NO_FIT
        sides = {side: f for side, f in fit["sides"].items() if f is not None}

        def used(dp, W):
            if not sides:  # not fitted yet: every sample with a pressure difference
                return dp != 0 and math.isfinite(W)
            side = "+" if dp > 0 else "-" if dp < 0 else None
            return side in sides and dp not in sides[side]["left_out"]

        rows = [r for r in self.rows if r["converged"] and key in r["W"]]
        kept = [(r["W"][key]["dp"], r["W"][key]["W"]) for r in rows if used(r["W"][key]["dp"], r["W"][key]["W"])]
        dp, W = np.array(kept, float).reshape(-1, 2).T
        # scale the axis to the samples that matter: just off dp = 0, W = dV/dp of a slack membrane is huge
        # (those points carry almost no weight in the |dp|-weighted fit); show them as markers at the top
        curves = [(np.linspace(*f["dp_range"], 200), side, f) for side, f in sides.items()]
        curves = [(x, side, f, np.polynomial.polynomial.polyval(x, f["coefficients"])) for x, side, f in curves]
        main = np.abs(dp) >= 0.05 * np.abs(dp).max() if len(dp) else np.array([], bool)
        shown = np.concatenate([W[main]] + [y for *_, y in curves]) if main.any() or curves else W
        top = None
        if len(shown) and np.isfinite(shown).all():
            lo, hi = shown.min(), shown.max()
            pad = 0.15 * (hi - lo or abs(hi) or 1.0)
            top = hi + pad
            ax.set_ylim(min(lo - pad, 0.0) if lo >= 0 else lo - pad, top)
        within = W <= top if top is not None else np.ones_like(W, bool)
        ax.plot(dp[within], W[within], "o", color="#2a6fdb", label="sampled W = dV/dp", zorder=3)
        if (~within).any():
            ax.plot(dp[~within], np.full((~within).sum(), top), "^", color="#2a6fdb", markerfacecolor="white",
                    clip_on=False, zorder=3, label="sampled, off scale (dp ≈ 0)")
        for x, side, f, y in curves:
            label = f"{SIDE_NAME[side]}: degree {f['degree']}" if len(sides) == 2 else f"fit: degree {f['degree']}"
            ax.plot(x, y, "-", color=SIDE_COLOR[side], label=label)
        if len(sides) == 2:
            ax.axvline(0.0, color="#cccccc", linewidth=0.8)
        left_out = {r["W"][key]["dp"] for r in rows if not used(r["W"][key]["dp"], r["W"][key]["W"])}
        for n, x in enumerate(sorted(left_out)):
            ax.axvline(x, color="#999999", linestyle=":", label="left out (dp = 0)" if n == 0 else None)
        activation = self.main.project.parts[self.activation_index].name
        ax.set_xlabel(f"p_a - p_0 [kPa]" if key == FLUID else f"p({key}) - p({activation}) [kPa]")
        ax.set_ylabel(f"W {self._weight_label(key)} [mm3/kPa]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # -----------------------------
    # Neuron equation
    # -----------------------------

    def equation(self):
        """The last equation fit, or None before one has finished (see _refit)."""
        return self._equation

    def _fits(self):
        fit = self.equation()
        return {k: fit["fits"].get(k, NO_FIT) for k in self.weight_keys} if fit else {}

    def _refit(self, then=None):
        """Fit the equation to the stored sweep rows in the background (no new simulations);
        `then(text)` is called with a status text when it is done."""
        if not (self.weight_keys and self.rows) or self.fit_worker is not None:
            return
        tolerance, method = self.tolerance.value(), self.method.currentData()
        self.fit_worker = Worker(lambda worker: equation_fit(self.rows, self.activation_index, tolerance, method))
        self.regenerate.setEnabled(False)
        self.status.setText(f"Fitting the equation to {tolerance:.4g} kPa ({self.method.currentText()})…")

        def succeeded(result):
            self.fit_worker = None
            self._equation = result
            self._summarise()
            self._plot()
            self.tabs.setCurrentIndex(1)
            self.regenerate.setEnabled(self.worker is None)
            text = (f"Equation for {tolerance:.4g} kPa: error on p_a {result['error']:.3g} kPa, total order "
                    f"{sum(weight_degree(f) for f in result['fits'].values())}.")
            (then or self.status.setText)(text)

        def failed(text):
            self.fit_worker = None
            self.regenerate.setEnabled(self.worker is None)
            self.status.setText("Equation fit failed: " + text.splitlines()[0])
            self.main.log_message(text)

        self.fit_worker.succeeded.connect(succeeded)
        self.fit_worker.failed.connect(failed)
        self.fit_worker.start()

    def equation_latex(self):
        """The neuron equation (neuron_equation_latex() pieces), or None without weights."""
        if not self.weight_keys or not self.rows:
            return None
        fits = self._fits()
        inputs = [(k, fits[k]) for k in self.weight_keys if k != FLUID]
        chamber = None
        if FLUID in fits and fits[FLUID]["kind"] != "none":
            p0 = next(r["W"][FLUID]["p"] for r in self.rows if FLUID in r["W"])
            chamber = (fits[FLUID], p0)
        activation = self.main.project.parts[self.activation_index].name
        return neuron_equation_latex(inputs, chamber, activation)

    def _summarise(self):
        equation = self.equation_latex()
        self.eq_figure.clear()
        if equation is None:
            self.eq_canvas.draw_idle()
            return
        # rendered: the main equation large, then each weight's sides, then its definition indented;
        # a weight's lines are clickable
        keys = self._equation_keys()
        lines = equation_lines(equation)
        if self.selected_weight not in keys:
            self.selected_weight = keys[0] if keys else None
        layout = {0: (0.02, 14), 1: (0.02, 10), 2: (0.08, 9)}
        for k, (line, n, level) in enumerate(lines):
            key = keys[n] if n is not None else None
            x, size = layout[level] if n is not None or level == 0 else (0.02, 9)
            text = self.eq_figure.text(x, 1 - (k + 0.8) / (len(lines) + 0.5), f"${line}$", fontsize=size,
                                       va="center", picker=key is not None,
                                       color="#d9480f" if key is not None and key == self.selected_weight else "black")
            text.weight_key = key
        self.eq_canvas.draw_idle()
        self._show_weight()
        # source: an align block plus how well it reproduces the sweep, and every weight, as comments
        fit, fits = self.equation(), self._fits()
        verdict = "within" if fit["met"] else "NOT within (every weight is at its highest useful degree)"
        notes = [f"% Solved for p_a, the equation reproduces all {len(fit['errors'])} sweep points to "
                 f"{fit['error']:.3g} kPa, {verdict} the tolerance of {fit['tolerance']:.4g} kPa.",
                 f"% Degrees by {fit['method']}, one polynomial per sign of dp: total order "
                 f"{sum(weight_degree(f) for f in fit['fits'].values())}, {fit['evaluated']} combinations tried."]

        def own(k):
            errors = fit["weight_errors"].get(k, {})
            return ("; alone it puts p_a off by " + ", ".join(f"{e:.3g} kPa ({SIDE_NAME[side]})"
                                                             for side, e in errors.items())) if errors else ""
        notes += [f"% W_{j} ({k}, path through {', '.join(self._shells(k))}): {describe_fit(fits[k])}{own(k)}"
                  for j, k in enumerate([k for k in self.weight_keys if k != FLUID], start=1)]
        if FLUID in fits:
            notes.append(f"% W_0 ({self._weight_label(FLUID)}): {describe_fit(fits[FLUID])}{own(FLUID)}")
        self.latex.setPlainText("\n".join(notes) + "\n" + equation_align(equation) + f"% {equation['note']}\n")

    def _equation_keys(self):
        """Weights in the order of the equation's lines: the inputs, then W0 when it is in the equation."""
        fits = self._fits()
        keys = [k for k in self.weight_keys if k != FLUID]
        return keys + ([FLUID] if FLUID in fits and fits[FLUID]["kind"] != "none" else [])

    def _pick_weight(self, event):
        key = getattr(event.artist, "weight_key", None)
        if key is not None and key != self.selected_weight:
            self.selected_weight = key
            for text in self.eq_figure.texts:
                if getattr(text, "weight_key", None) is not None:
                    text.set_color("#d9480f" if text.weight_key == key else "black")
            self.eq_canvas.draw_idle()
            self._show_weight()

    def _show_weight(self):
        """The selected weight's fitted polynomial over its sampled points, under the equation."""
        self.weight_figure.clear()
        key = self.selected_weight
        if key is not None and self.equation():
            ax = self.weight_figure.add_subplot(111)
            self._plot_fit(ax, key)
            keys = self._equation_keys()
            name = "W_0" if key == FLUID else f"W_{keys.index(key) + 1}"
            own = self.equation()["weight_errors"].get(key, {})
            off = ", ".join(f"{e:.3g} kPa ({SIDE_NAME[side]})" for side, e in own.items())
            ax.set_title(f"{name}: {self._weight_label(key)}" + (f"  (alone puts p_a off by {off})" if off else ""),
                         fontsize=9)
        self.weight_canvas.draw_idle()

    def _shells(self, key):
        return next((r["W"][key]["shells"] for r in self.rows if key in r["W"]), [])

    def _done(self, mesh_data):
        self.main.adopt_mesh(mesh_data)
        text = f"Sweep finished: {len(self.rows)} points, {sum(r['time'] for r in self.rows):.0f} s."
        self._finish(text)
        self._refit(then=lambda fitted: self.status.setText(f"{text} {fitted}"))

    def _failed(self, text):
        self._finish("Sweep stopped: " + text.splitlines()[0])
        if text != "Cancelled.":
            self.main.log_message(text)

    def _finish(self, text):
        self.status.setText(text)
        self.regenerate.setEnabled(bool(self.weight_keys and self.rows) and self.fit_worker is None)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    # -----------------------------
    # Export
    # -----------------------------

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
            header += [f"dV {parts[c].name} [mm3]" for c in self.chambers]
            if self.weight_keys:
                header.append("p_a from weights [kPa]")
            header += [f"{name} {self._weight_label(k)} [{unit}]"
                       for k in self.weight_keys for name, unit, _ in WEIGHT_FIELDS.values()]
            writer.writerow(header + ["converged"])
            for r in sorted(self.rows, key=lambda r: (r["a"], r["b"] if r["b"] is not None else 0)):
                writer.writerow([r["a"]] + ([r["b"]] if self.b_index is not None else [])
                                + [r["P"][c] for c in self.chambers] + [r["dV"][c] for c in self.chambers]
                                + ([r["p_a_rebuilt"]] if self.weight_keys else [])
                                + [r["W"].get(k, {}).get(field, math.nan)
                                   for k in self.weight_keys for field in WEIGHT_FIELDS]
                                + [r["converged"]])
        written = [path]
        if self.weight_keys and self.equation() is None:
            self._equation = equation_fit(self.rows, self.activation_index, self.tolerance.value(),
                                          self.method.currentData())
            self._summarise()
        if self.weight_keys:
            base = os.path.splitext(path)[0]
            self._export_weights(base + "_weights.csv")
            with open(base + "_equation.tex", "w", encoding="utf-8") as f:
                f.write(self.latex.toPlainText())
            written += [base + "_weights.csv", base + "_equation.tex"]
        self.status.setText("Exported " + ", ".join(os.path.basename(p) for p in written) + ".")

    def _export_weights(self, path):
        """One line per weight and side of dp: its W(dp) polynomial, and the equation's error on p_a."""
        fit, fits = self.equation(), self._fits()
        activation = self.main.project.parts[self.activation_index].name
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["weight", "input", "dp", "side", "membranes on the path", "degree",
                             "W(dp) coefficients c0 c1 c2 ... [mm3/kPa, dp in kPa]", "points",
                             "dp min [kPa]", "dp max [kPa]", "equation error on p_a [kPa]",
                             "equation tolerance [kPa]"])
            inputs = [k for k in self.weight_keys if k != FLUID]
            for name, k in [(f"W_{j}", k) for j, k in enumerate(inputs, start=1)] + \
                           ([("W_0", FLUID)] if FLUID in fits else []):
                dp = f"p_a - p_0 ({activation})" if k == FLUID else f"p({k}) - p({activation})"
                sides = {side: w for side, w in fits[k]["sides"].items() if w is not None}
                for side, w in sides.items():
                    applies = SIDE_NAME[side] if len(sides) == 2 else f"all dp (sampled only {SIDE_NAME[side]})"
                    writer.writerow([name, self._weight_label(k), dp, applies, " ".join(self._shells(k)),
                                     w["degree"], " ".join(f"{c:.10g}" for c in w["coefficients"]), w["points"],
                                     *w["dp_range"], fit["error"], fit["tolerance"]])

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(30000)
        if self.fit_worker is not None:
            self.fit_worker.wait(60000)
        super().closeEvent(event)
