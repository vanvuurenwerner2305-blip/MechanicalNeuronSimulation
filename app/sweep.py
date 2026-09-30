"""Parameter sweep over one or two input chamber pressures (the neuron's response surface).

Besides the chamber pressures, every point records the mechanical weight of every input path into
the chosen pre-activation chamber: W_j = dV_j / (p_j - p_a), with dV_j the volume the path pushes into
the pre-activation chamber (see membrane_sim.characterise; a path lumps its membranes and intermediate
chambers), plus the pre-activation chamber's own compliance W_0. After the sweep each W is fitted as
the lowest-degree polynomial W(dp) that passes within the tolerance of every point (degree 0: W is
constant; points with dp = 0 are left out), and the neuron equation is written out in LaTeX.
"""
import csv
import io
import math
import os
import time

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen, QPixmap
from qtpy.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                            QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                            QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                            QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from membrane_sim.characterise import (BIGGEST_ERROR, LOWEST_TOTAL, activation_sensitivities, bias_volumes,
                                       fit_neuron_equation, input_paths, input_weights, is_neutral,
                                       solve_activation, weight_coefficients, equation_align, neuron_equation_latex, polynomial_text,
                                       rebuild_activation_pressure, weight_degree)

from .builder import KPA, build_environment, generate_mesh
from .project import ACTIVATION_MEMBRANE, CHAMBER, CONSTANT, VENT
from .workers import Worker

FLUID = "W0"  # key of the pre-activation chamber's own compliance among the weights
# per-path quantities: key -> (label, unit, factor from model units: mm, MPa)
WEIGHT_FIELDS = {
    "p": ("p", "kPa", 1.0 / KPA),
    "dp": ("dp", "kPa", 1.0 / KPA),
    "dV": ("dV", "mm3", 1.0),
    "W": ("W", "mm3/kPa", KPA),
    "W_tan": ("dV/dp tangent", "mm3/kPa", KPA),
}


# quantities of an activation membrane (a linked activation-function design) per sweep point, besides its named
# outputs (keys "out:<name>", kPa)
ACTIVATION_QUANTITIES = {
    "dp": ("pre-activation Δp", "kPa"),
    "area": ("tube area", "mm2"),
    "mdot": ("mass flow", "kg/s"),
}


def activation_row(out) -> dict:
    """A design's outputs at one point (ActivationDesign.outputs) as flat sweep values: named outputs first."""
    row = {f"out:{name}": v for name, v in out["outputs"].items()}
    row.update({k: out[k] for k in ACTIVATION_QUANTITIES})
    return row


def activation_label(key):
    """(name, unit) of a flat activation value."""
    return (key[4:], "kPa") if key.startswith("out:") else ACTIVATION_QUANTITIES[key]


def point_weights(build, activation_index, load_factor):
    """Weights of every input path (label -> {key: value in WEIGHT_FIELDS units, "shells": names}),
    the pre-activation chamber's own term under FLUID, and the pre-activation pressure rebuilt from them."""
    raw = input_weights(build.env, build.volumes[activation_index], load_factor)
    scale = lambda w: {k: w[k] * f for k, (_, _, f) in WEIGHT_FIELDS.items()}
    weights = {w["input"]: {**scale(w), "shells": w["shells"], "biased": w.get("biased", False),
                            "bias_chambers": w.get("bias_chambers", [])} for w in raw["inputs"]}
    weights[FLUID] = {**scale(raw["chamber"]), "shells": []}
    return weights, rebuild_activation_pressure(raw) / KPA


def activation_value(row, index, field):
    """An activation membrane's output at a sweep point (nan when it was not built)."""
    return row.get("act", {}).get(index, {}).get(field, math.nan)


def sample_weight(row, key):
    """W of a weight at a sweep point: the secant dV / dp, or for a biased path (dV - b) / dp with its
    measured bias b (row["bias"], see run_sweep)."""
    w = row["W"][key]
    b = row.get("bias", {}).get(key)
    if b is None:
        return w["W"]
    return (w["dV"] - b) / w["dp"] if w["dp"] != 0 and math.isfinite(w["dp"]) else math.nan


def weight_sensitivity(row, key):
    """dp_a/dW of weight `key` at a sweep point (Article 2, Eq. 4.10): (p_k - p_a) / sum of all W, in
    kPa per mm3/kPa. For the pre-activation fluid's own W_0 the pressure difference is p_0 - p_a = -dp."""
    return activation_sensitivities({k: (w["dp"], sample_weight(row, k)) for k, w in row["W"].items()})[key]


def equation_fit(rows, activation_index, tolerance, method=LOWEST_TOTAL, check=None):
    """Fit the weights of the converged rows so the equation meets `tolerance` (kPa) on p_a.
    check() is called between degree combinations and may raise to cancel (Worker.check)."""
    rows = [r for r in rows if r["converged"] and r["W"]]
    samples = [{"p_a": r["P"][activation_index],
                "terms": {k: (w["p"], w["dp"], w["W"]) for k, w in r["W"].items()},
                "volumes": {k: w.get("dV", math.nan) for k, w in r["W"].items()}} for r in rows]
    bias = rows[0].get("bias", {}) if rows else {}
    return fit_neuron_equation(samples, tolerance, method, check=check, bias=bias)


def measure_bias(build, project, activation_index, callback, log):
    """{path label: b_j (mm3)} of the biased paths into the pre-activation chamber, from one solve at the
    neutral state: every constant-pressure input at the pre-activation chamber's rest pressure p_0. There an
    unbiased path pushes nothing; a path through a chamber that is not neutral at rest (e.g. a gas weight
    chamber filled above ambient) pushes its bias. {} when no path is biased (no extra solve)."""
    activation = build.volumes[activation_index]
    volumes = list(build.volumes.values())
    biased = [label for label, _, _, between in input_paths(activation, volumes)
              if any(not is_neutral(v) for v in between)]
    if not biased:
        return {}
    inputs = [v for i, v in build.volumes.items() if project.parts[i].props.get("model") == CONSTANT]  # not vents
    saved = {id(v): v.P0 for v in inputs}
    p0 = activation.pressure(0.0)
    log(f"Measuring the bias of {', '.join(biased)} (all inputs at the pre-activation chamber's rest pressure)…")
    try:
        for v in inputs:
            v.P0 = p0
        result = build.solve(project.solver, callback, fixed_contact=True)
        if not result.converged:
            log("The bias could not be measured (the neutral state did not converge); fitting without it.")
            return {}
        return bias_volumes(input_weights(build.env, activation, result.load_factor))
    finally:
        for v in inputs:
            v.P0 = saved[id(v)]


NO_FIT = {"kind": "none", "sides": {"+": None, "-": None}}
SIDE_NAME = {"+": "dp > 0", "-": "dp < 0"}
SIDE_COLOR = {"+": "#d9480f", "-": "#7048e8"}


def describe_side(fit):
    if fit["degree"] == 0:
        return f"W = {fit['coefficients'][0]:.6g} mm3/kPa, constant ({fit['points']} points)"
    return (f"W(dp) = {polynomial_text(fit['coefficients'], precision=6)}, degree {fit['degree']} "
            f"({fit['points']} points)")


def math_pixmap(latex, size=12, color="black", ratio=1.0):
    """A matplotlib-mathtext line rendered at its natural size (sharp on high-DPI screens)."""
    buffer = io.BytesIO()
    math_to_image(f"${latex}$", buffer, prop=FontProperties(size=size), dpi=100 * ratio, format="png",
                  color=color)
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def math_label(latex, size=12, color="black", ratio=1.0):
    label = QLabel()
    label.setPixmap(math_pixmap(latex, size, color, ratio))
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return label


class Brace(QWidget):
    """A left curly brace as tall as the rows it groups (mathtext has no cases environment)."""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = QColor(color)
        self.setFixedWidth(14)

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        x0, x1, m = w - 2.0, 4.0, h / 2.0
        path = QPainterPath()
        path.moveTo(x0, 2)
        path.cubicTo(x1, 2, x0, m, x1 - 3, m)
        path.cubicTo(x0, m, x1, h - 2, x0, h - 2)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self.color, 1.4))
        painter.drawPath(path)


class WeightCard(QFrame):
    """One weight of the neuron equation: its polynomial(s), definition and fit quality; click to plot it."""

    def __init__(self, key, title, weight, info, ratio, colors, on_click, parent=None):
        super().__init__(parent)
        self.key, self.on_click, self.colors = key, on_click, colors
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to plot this weight's fit over its sampled points.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        heading = QLabel(title)
        heading.setTextFormat(Qt.RichText)
        layout.addWidget(heading)

        body = QGridLayout()
        body.setHorizontalSpacing(4)
        body.setVerticalSpacing(2)
        pieces = weight["pieces"]
        body.addWidget(math_label(rf"{weight['symbol']}({weight['variable']}) =", 13, colors["text"], ratio),
                       0, 0, len(pieces), 1)
        column = 1
        if len(pieces) > 1:
            body.addWidget(Brace(colors["text"]), 0, 1, len(pieces), 1)
            column = 2
        for row, (poly, condition) in enumerate(pieces):
            text = poly + (rf",\quad {condition}" if condition else "")
            body.addWidget(math_label(text, 13, colors["text"], ratio), row, column)
        body.setColumnStretch(column + 1, 1)
        layout.addLayout(body)
        layout.addWidget(math_label(weight["definition"], 11, colors["muted"], ratio))
        details = QLabel(info)
        details.setWordWrap(True)
        details.setStyleSheet(f"color: {colors['muted']};")
        layout.addWidget(details)
        self.set_selected(False)

    def set_selected(self, selected):
        border = self.colors["accent"] if selected else self.colors["border"]
        width = 2 if selected else 1
        self.setStyleSheet(f"WeightCard {{ border: {width}px solid {border}; border-radius: 6px; "
                           f"background: {self.colors['card']}; }}")

    def mousePressEvent(self, event):
        self.on_click(self.key)
        super().mousePressEvent(event)


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
                         "it can not be the pre-activation chamber.")
    # one contact stiffness for the whole sweep, sized for its highest pressure
    p_max = max([abs(v.P0) for v in build.volumes.values()] + [abs(x) * KPA for x in a_values]
                + ([abs(x) * KPA for x in b_values] if b_index is not None else []))
    build.set_contact_stiffness(p_max)
    points = sweep_points(list(a_values), list(b_values) if b_index is not None else [None])
    bias = {}
    if activation_index is not None:
        bias = measure_bias(build, project, activation_index, lambda lam, it, r: worker.check(), worker.log)
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
               "W": {}, "p_a_rebuilt": math.nan,
               "act": {i: activation_row(out) for i, out in build.activation_outputs().items()},
               "extrapolated": build.range_warnings(project.parts)}
        for w in row["extrapolated"]:
            worker.log("Warning: " + w)
        if activation_index is not None:
            row["W"], row["p_a_rebuilt"] = point_weights(build, activation_index, result.load_factor)
            row["bias"] = bias
        worker.item.emit(row)
        worker.report((k + 1) / len(points), f"Sweep point {k + 1}/{len(points)}")
    return mesh_data


class SweepDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Parameter sweep")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        self.resize(min(1500, int(0.9 * available.width())) if available else 1400,
                    min(950, int(0.9 * available.height())) if available else 900)
        self.worker = None
        self.rows = []
        self.weight_keys = []
        self.activation_index = None
        self._equation = None    # equation_fit of the current rows (set by the fit worker)
        self.fit_worker = None   # background equation fit (the exhaustive search can take seconds)
        self._stopping = []      # cancelled fits whose threads have not finished yet
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

        grid.addWidget(QLabel("Pre-activation chamber"), 3, 0)
        self.activation = QComboBox()
        self.activation.addItem("(none: no weights)", None)
        for i in self.closed:
            self.activation.addItem(parts[i].name, i)
        named = [k for k, i in enumerate(self.closed) if "activ" in parts[i].name.lower()]
        self.activation.setCurrentIndex(1 + (named[0] if named else 0) if self.closed else 0)
        self.activation.setToolTip("The chamber whose pressure p_a is the neuron's pre-activation. Every input path "
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
        # Neuron equation: a verdict banner, then the equation (scrollable, at its natural size, one card per
        # weight) next to the selected weight's fit plot
        equation = QWidget()
        el = QVBoxLayout(equation)
        self.eq_summary = QLabel("The neuron equation appears here after a sweep with an pre-activation chamber.")
        self.eq_summary.setWordWrap(True)
        self.eq_summary.setTextFormat(Qt.RichText)
        self.eq_summary.setContentsMargins(8, 6, 8, 6)
        el.addWidget(self.eq_summary)
        eq_split = QSplitter(Qt.Horizontal)
        self.eq_scroll = QScrollArea()
        self.eq_scroll.setWidgetResizable(True)
        self.eq_scroll.setFrameShape(QFrame.NoFrame)
        self.eq_body = None
        self.weight_cards = {}
        eq_split.addWidget(self.eq_scroll)
        weight_panel = QWidget()
        wl = QVBoxLayout(weight_panel)
        wl.setContentsMargins(0, 0, 0, 0)
        self.weight_figure = Figure(figsize=(6, 4.5), tight_layout=True)
        self.weight_canvas = FigureCanvasQTAgg(self.weight_figure)
        wl.addWidget(NavigationToolbar2QT(self.weight_canvas, weight_panel))
        wl.addWidget(self.weight_canvas, 1)
        eq_split.addWidget(weight_panel)
        eq_split.setStretchFactor(0, 3)
        eq_split.setStretchFactor(1, 2)
        eq_split.setSizes([600, 420])
        el.addWidget(eq_split, 1)
        self.selected_weight = None  # key of the weight whose fit is plotted
        self.tabs.addTab(equation, "Neuron equation")

        compare = QWidget()
        cl = QVBoxLayout(compare)
        self.compare_figure = Figure(figsize=(6, 5), tight_layout=True)
        self.compare_canvas = FigureCanvasQTAgg(self.compare_figure)
        cl.addWidget(NavigationToolbar2QT(self.compare_canvas, compare))
        cl.addWidget(self.compare_canvas, 1)
        self.tabs.addTab(compare, "Equation vs simulation")

        source = QWidget()
        sl = QVBoxLayout(source)
        self.latex = QPlainTextEdit()
        self.latex.setReadOnly(True)
        self.latex.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.latex.setPlaceholderText("The neuron equation (LaTeX) appears here after a sweep with an pre-activation chamber.")
        sl.addWidget(self.latex, 1)
        copy = QPushButton("Copy LaTeX")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.latex.toPlainText()))
        sl.addWidget(copy, 0, Qt.AlignRight)
        self.tabs.addTab(source, "LaTeX")
        self._clear_equation()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([420, 1080])
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
        for i in self.linked:
            for field in self._activation_keys(i):
                name, unit = activation_label(field)
                self.output.addItem(f"{parts[i].name}: {name} [{unit}]", ("act", (i, field)))
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
        self._cancel_fit()
        self.activation_index = self.activation.currentData()
        self.rows, self.weight_keys = [], []
        self._fill_outputs()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.latex.clear()
        self._clear_equation()

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
        headers += self._activation_headers()
        if self.weight_keys:
            headers.append("p_a from weights [kPa]")
        headers += [f"W {self._weight_label(k)} [mm3/kPa]" for k in self.weight_keys]
        return headers + ["Converged", "Time [s]"]

    def _activation_keys(self, i):
        """The values of linked part i, from the first sweep point (its design's outputs are known then)."""
        first = self.rows[0].get("act", {}).get(i) if self.rows else None
        return list(first) if first else list(ACTIVATION_QUANTITIES)

    def _activation_headers(self):
        parts = self.main.project.parts
        return [f"{parts[i].name} {'{} [{}]'.format(*activation_label(k))}" for i in self.linked
                for k in self._activation_keys(i)]

    def _activation_values(self, row):
        return [activation_value(row, i, k) for i in self.linked for k in self._activation_keys(i)]

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
        values += self._activation_values(row)
        if self.weight_keys:
            values.append(row["p_a_rebuilt"])
        values += [row["W"].get(k, {}).get("W", math.nan) for k in self.weight_keys]
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
        if field == "fit":
            self._plot_fit(ax, key)
            self.canvas.draw_idle()
            return
        if field == "P":
            value, label = (lambda r: r["P"][key]), f"{parts[key].name} pressure [kPa]"
        elif field == "act":
            name, unit = activation_label(key[1])
            value, label = (lambda r: activation_value(r, *key)), f"{parts[key[0]].name}: {name} [{unit}]"
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
        kept = [(r["W"][key]["dp"], sample_weight(r, key), abs(weight_sensitivity(r, key)))
                for r in rows if used(r["W"][key]["dp"], sample_weight(r, key))]
        dp, W, sensitivity = np.array(kept, float).reshape(-1, 3).T
        # scale the axis to the samples that matter: just off dp = 0, W = dV/dp of a slack membrane is huge
        # (those points carry almost no weight in the sensitivity-weighted fit); show them as markers at the top
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
        # dots coloured by how much p_a responds to this weight at that point, |dp_a/dW| (Eq. 4.10)
        finite = np.isfinite(sensitivity)
        colours = dict(cmap="viridis", vmin=sensitivity[finite].min() if finite.any() else 0.0,
                       vmax=sensitivity[finite].max() if finite.any() else 1.0, edgecolors="#333333",
                       linewidths=0.5, zorder=3)
        biased = key in next((r.get("bias", {}) for r in rows), {})
        dots = ax.scatter(dp[within], W[within], c=sensitivity[within], marker="o",
                          label="sampled W = (dV - b)/dp" if biased else "sampled W = dV/dp", **colours)
        if (~within).any():
            ax.scatter(dp[~within], np.full((~within).sum(), top), c=sensitivity[~within], marker="^",
                       clip_on=False, label="sampled, off scale (dp ≈ 0)", **colours)
        if len(dp):
            ax.figure.colorbar(dots, ax=ax, label="|dp_a/dW| [kPa per mm3/kPa]")
        for x, side, f, y in curves:
            label = f"{SIDE_NAME[side]}: degree {f['degree']}" if len(sides) == 2 else f"fit: degree {f['degree']}"
            ax.plot(x, y, "-", color=SIDE_COLOR[side], label=label)
        if len(sides) == 2:
            ax.axvline(0.0, color="#cccccc", linewidth=0.8)
        left_out = {r["W"][key]["dp"] for r in rows if not used(r["W"][key]["dp"], sample_weight(r, key))}
        for n, x in enumerate(sorted(left_out)):
            ax.axvline(x, color="#999999", linestyle=":", label="left out (dp ≈ 0, or a negative W at that point)" if n == 0 else None)
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
        if self._closing:
            return
        tolerance, method = self.tolerance.value(), self.method.currentData()
        rows, activation = list(self.rows), self.activation_index
        job = self.fit_worker = Worker(lambda worker: equation_fit(rows, activation, tolerance, method,
                                                                   check=worker.check))
        self.regenerate.setEnabled(False)
        self.status.setText(f"Fitting the equation to {tolerance:.4g} kPa ({self.method.currentText()})…")

        def succeeded(result):
            if self.fit_worker is not job:  # cancelled, or replaced by a new sweep
                return
            self.fit_worker = None
            self._equation = result
            self._summarise()
            self._plot()
            self.tabs.setCurrentIndex(1)
            self.regenerate.setEnabled(self.worker is None)
            text = (f"Equation for {tolerance:.4g} kPa: error on p_a {result['error']:.3g} kPa, total order "
                    f"{sum(weight_degree(f) for f in result['fits'].values())}.")
            if result.get("stopped"):
                text += f" Search stopped after {result['evaluated']} combinations (best found shown)."
            (then or self.status.setText)(text)

        def failed(text):
            if self.fit_worker is not job:
                return
            self.fit_worker = None
            self.regenerate.setEnabled(self.worker is None)
            self.status.setText("Equation fit failed: " + text.splitlines()[0])
            self.main.log_message(text)

        self.fit_worker.succeeded.connect(succeeded)
        self.fit_worker.failed.connect(failed)
        self.fit_worker.start()

    def equation_latex(self, precision=6):
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
        return neuron_equation_latex(inputs, chamber, activation, precision=precision)

    def _colors(self):
        palette = self.palette()
        text, base = palette.color(QPalette.Text), palette.color(QPalette.Base)
        dark = base.lightness() < 128
        return {"text": text.name(), "muted": "#9aa0a6" if dark else "#5f6368", "accent": "#d9480f",
                "border": "#3c4043" if dark else "#d0d4d9", "card": base.name(),
                "ok": "#1e4620" if dark else "#e6f4ea", "bad": "#5c1a1a" if dark else "#fce8e6"}

    def _clear_equation(self):
        self.weight_cards = {}
        self.eq_body = QWidget()
        self.eq_scroll.setWidget(self.eq_body)
        self.eq_summary.setText("The neuron equation appears here after a sweep with an pre-activation chamber.")
        self.eq_summary.setStyleSheet("")
        self.weight_figure.clear()
        self.weight_canvas.draw_idle()
        self.compare_figure.clear()
        self.compare_canvas.draw_idle()

    def _summarise(self):
        equation = self.equation_latex(precision=4)  # the rendered view is rounded, the LaTeX source is not
        fit = self.equation()
        if equation is None or fit is None:
            self._clear_equation()
            return
        keys = self._equation_keys()
        if self.selected_weight not in keys:
            self.selected_weight = keys[0] if keys else None
        self._render_equation(equation, fit, keys)
        self._show_weight()
        self._plot_compare()
        # source: an align block plus how well it reproduces the sweep, and every weight, as comments
        equation, fits = self.equation_latex(), self._fits()
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
        if fit.get("bias"):
            notes.append(f"% B = {fit['bias']:.6g} mm3: measured bias of the biased paths (every input at the "
                         "pre-activation chamber's rest pressure); their W = (dV - b) / dp.")
        self.latex.setPlainText("\n".join(notes) + "\n" + equation_align(equation) + f"% {equation['note']}\n")

    def _equation_keys(self):
        """Weights in the order of the equation's lines: the inputs, then W0 when it is in the equation."""
        fits = self._fits()
        keys = [k for k in self.weight_keys if k != FLUID]
        return keys + ([FLUID] if FLUID in fits and fits[FLUID]["kind"] != "none" else [])

    def _render_equation(self, equation, fit, keys):
        """The verdict banner, then (scrollable) the main equation and one clickable card per weight."""
        colors = self._colors()
        ratio = self.devicePixelRatioF()
        met = fit["met"]
        total = sum(weight_degree(f) for f in fit["fits"].values())
        verdict = "meets" if met else "does <b>not</b> meet"
        self.eq_summary.setText(
            f"Solved for p<sub>a</sub>, the equation reproduces all {len(fit['errors'])} sweep points to within "
            f"<b>{fit['error']:.3g} kPa</b>, and {verdict} the tolerance of {fit['tolerance']:.4g} kPa. "
            f"Total polynomial order {total} ({fit['method']}, {fit['evaluated']} combinations tried)."
            + (f" The search stopped after its budget of {fit['evaluated']} combinations: this is the best found, "
               "a looser tolerance gives a complete search." if fit.get("stopped") else
               "" if met else " Every weight is already at its highest useful degree."))
        self.eq_summary.setStyleSheet(f"background: {colors['ok' if met else 'bad']}; border-radius: 6px;")

        self.weight_cards = {}
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        heading = QLabel("<b>Pre-activation pressure p<sub>a</sub></b>")
        layout.addWidget(heading)
        layout.addWidget(math_label(equation["main"], 16, colors["text"], ratio))
        layout.addWidget(QLabel(f"<b>Weights</b> <span style='color:{colors['muted']}'>"
                                f"(W in mm³/kPa, pressures in kPa; click a weight to plot its fit)</span>"))
        for n, (key, weight) in enumerate(zip(keys, equation["weights"])):
            card = WeightCard(key, self._card_title(key, n), weight, self._card_info(key, fit), ratio, colors,
                              self._select_weight)
            card.set_selected(key == self.selected_weight)
            self.weight_cards[key] = card
            layout.addWidget(card)
        if equation.get("bias"):
            layout.addWidget(self._bias_card(equation["bias"], ratio, colors))
        layout.addStretch(1)
        self.eq_body = body
        self.eq_scroll.setWidget(body)

    def _bias_card(self, bias, ratio, colors):
        """The bias B: what the biased paths push into the pre-activation chamber at zero pressure difference."""
        card = QFrame()
        card.setObjectName("bias")
        card.setStyleSheet(f"QFrame#bias {{ border: 1px solid {colors['border']}; border-radius: 6px; "
                           f"background: {colors['card']}; }}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addWidget(QLabel("<b>B</b> &nbsp; bias"))
        layout.addWidget(math_label(bias["definition"], 13, colors["text"], ratio))
        chambers = {name for r in self.rows for w in r["W"].values() for name in w.get("bias_chambers", [])}
        text = ("Volume the biased paths push into the pre-activation chamber when every pressure is equal, "
                "because " + ", ".join(sorted(chambers)) + (" is" if len(chambers) == 1 else " are")
                + " not neutral at rest (filled above or below ambient). Measured by one extra solve with every "
                "input at the pre-activation chamber's rest pressure; not fitted.")
        details = QLabel(text)
        details.setWordWrap(True)
        details.setStyleSheet(f"color: {colors['muted']};")
        layout.addWidget(details)
        return card

    def _card_title(self, key, n):
        symbol = "W<sub>0</sub>" if key == FLUID else f"W<sub>{n + 1}</sub>"
        if key == FLUID:
            return f"<b>{symbol}</b> &nbsp; compliance of the pre-activation fluid ({self._weight_label(key)})"
        shells = ", ".join(self._shells(key))
        return f"<b>{symbol}</b> &nbsp; input <b>{key}</b>" + (f" &nbsp;·&nbsp; path through {shells}" if shells else "")

    def _card_info(self, key, fit):
        """Degree and points per side, and the p_a error this weight causes on its own."""
        parts = []
        b = fit["fits"].get(key, {}).get("bias")
        if b:
            parts.append(f"biased path: b = {b:.4g} mm³ (W from the volume minus b)")
        own = fit["weight_errors"].get(key, {})
        for side, f in fit["fits"].get(key, NO_FIT)["sides"].items():
            if f is None:
                continue
            text = f"{SIDE_NAME[side]}: degree {f['degree']}, {f['points']} points"
            if side in own:
                text += f", alone puts p_a off by {own[side]:.3g} kPa"
            parts.append(text)
        return "; ".join(parts) or "no points with a pressure difference"

    def _equation_activation(self, a, b=None):
        """Every p_a the fitted equation allows with input A at `a` and input B at `b` (kPa), sorted; every
        other pressure (inputs not swept, the ambient, p_0) as in the sweep."""
        fit = self.equation()
        parts = self.main.project.parts
        swept = {parts[self.a_index].name: a}
        if self.b_index is not None:
            swept[parts[self.b_index].name] = b
        row = next(r for r in self.rows if r["converged"] and r["W"])
        terms = []
        for k in self._equation_keys():
            p = swept.get(k, row["W"][k]["p"]) if k != FLUID else row["W"][k]["p"]
            if math.isfinite(p):
                terms.append((p, weight_coefficients(fit["fits"][k]), k == FLUID))
        return solve_activation(terms, bias=fit.get("bias", 0.0), all_roots=True) if terms else [math.nan]

    def _plot_compare(self):
        """The simulated pre-activation pressure with the fitted equation's prediction over it: points and a
        curve for a one-input sweep, surfaces for a two-input sweep (the equation on a finer grid, so it
        also shows what it does between the sweep points)."""
        self.compare_figure.clear()
        fit = self.equation()
        rows = [r for r in self.rows if r["converged"] and r["W"]]
        if fit is None or not rows:
            self.compare_canvas.draw_idle()
            return
        parts = self.main.project.parts
        name_a, activation = parts[self.a_index].name, parts[self.activation_index].name
        accent, title = "#d9480f", (f"p_a of {activation}: the equation is within {fit['error']:.3g} kPa of the "
                                    f"simulation at the sweep points (tolerance {fit['tolerance']:.4g} kPa)")
        pick = lambda roots, near: min(roots, key=lambda x: abs(x - near))  # the branch that follows the sim
        if self.b_index is None:
            ax = self.compare_figure.add_subplot(111)
            rows = sorted(rows, key=lambda r: r["a"])
            a = np.array([r["a"] for r in rows])
            simulated = np.array([r["P"][self.activation_index] for r in rows])
            fine = np.linspace(a.min(), a.max(), 200)
            near = np.interp(fine, a, simulated)
            roots = [self._equation_activation(x) for x in fine]
            ax.plot(fine, [pick(rs, y) for rs, y in zip(roots, near)], "-", color=accent, label="equation")
            others = [(x, r) for x, rs, y in zip(fine, roots, near) for r in rs if r != pick(rs, y)]
            if others:
                ax.plot(*zip(*others), "x", color="#c92a2a", markersize=4,
                        label="other solutions of the equation (it is multivalued there)")
            ax.plot(a, simulated, "o", color="#2a6fdb", label="simulated")
            ax.set_xlabel(f"{name_a} [kPa]")
            ax.set_ylabel("p_a [kPa]")
            ax.grid(alpha=0.3)
            ax.legend()
        else:
            name_b = parts[self.b_index].name
            ax = self.compare_figure.add_subplot(111, projection="3d")
            simulated = np.full((len(self.b_values), len(self.a_values)), np.nan)
            for r in rows:
                simulated[r["j"], r["i"]] = r["P"][self.activation_index]
            A, B = np.meshgrid(self.a_values, self.b_values)
            if len(self.a_values) > 1 and len(self.b_values) > 1:
                ax.plot_surface(A, B, simulated, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True)
            ax.scatter(A.ravel(), B.ravel(), simulated.ravel(), color="#2a6fdb", s=12, depthshade=False)
            n = 25
            fa = np.linspace(self.a_values.min(), self.a_values.max(), n if len(self.a_values) > 1 else 1)
            fb = np.linspace(self.b_values.min(), self.b_values.max(), n if len(self.b_values) > 1 else 1)
            FA, FB = np.meshgrid(fa, fb)
            roots = [[self._equation_activation(x, y) for x in fa] for y in fb]
            nearest_sim = lambda x, y: simulated[np.nanargmin(np.abs(self.b_values - y)), np.nanargmin(np.abs(self.a_values - x))]
            predicted = np.array([[pick(roots[j][i], nearest_sim(fa[i], fb[j])) for i in range(len(fa))]
                                  for j in range(len(fb))])
            ax.plot_wireframe(FA, FB, predicted, color=accent, linewidth=0.7)
            ambiguous = sum(len(rs) > 1 for row in roots for rs in row)
            if ambiguous:
                title += f"\n{ambiguous} of {FA.size} grid points have more than one solution (nearest to the simulation shown)"
            ax.set_xlabel(f"{name_a} [kPa]")
            ax.set_ylabel(f"{name_b} [kPa]")
            ax.set_zlabel("p_a [kPa]")
            ax.legend(handles=[Line2D([], [], color="#2a6fdb", marker="o", linestyle="", label="simulated (surface)"),
                               Line2D([], [], color=accent, label="equation (wireframe)")], loc="upper left")
        ax.set_title(title, fontsize=9)
        self.compare_canvas.draw_idle()

    def _select_weight(self, key):
        if key == self.selected_weight:
            return
        self.selected_weight = key
        for k, card in self.weight_cards.items():
            card.set_selected(k == key)
        self._show_weight()

    def _show_weight(self):
        """The selected weight's fitted polynomial over its sampled points, next to the equation."""
        self.weight_figure.clear()
        key = self.selected_weight
        if key is not None and self.equation():
            ax = self.weight_figure.add_subplot(111)
            self._plot_fit(ax, key)
            keys = self._equation_keys()
            name = "W_0" if key == FLUID else f"W_{keys.index(key) + 1}"
            ax.set_title(f"{name}: {self._weight_label(key)}", fontsize=10)
        self.weight_canvas.draw_idle()

    def _shells(self, key):
        return next((r["W"][key]["shells"] for r in self.rows if key in r["W"]), [])

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
            header += self._activation_headers()
            if self.weight_keys:
                header.append("p_a from weights [kPa]")
            header += [f"{name} {self._weight_label(k)} [{unit}]"
                       for k in self.weight_keys for name, unit, _ in WEIGHT_FIELDS.values()]
            writer.writerow(header + ["converged"])
            for r in sorted(self.rows, key=lambda r: (r["a"], r["b"] if r["b"] is not None else 0)):
                writer.writerow([r["a"]] + ([r["b"]] if self.b_index is not None else [])
                                + [r["P"][c] for c in self.chambers] + [r["dV"][c] for c in self.chambers]
                                + self._activation_values(r)
                                + ([r["p_a_rebuilt"]] if self.weight_keys else [])
                                + [r["W"].get(k, {}).get(field, math.nan)
                                   for k in self.weight_keys for field in WEIGHT_FIELDS]
                                + [r["converged"]])
        written = [path]
        note = ""
        if self.weight_keys and self.equation() is None:
            # fitting can take a while: never on the UI thread; export the points now, the equation later
            note = " The equation is not fitted yet; export again once it is."
            self._refit()
        elif self.weight_keys:
            base = os.path.splitext(path)[0]
            self._export_weights(base + "_weights.csv")
            with open(base + "_equation.tex", "w", encoding="utf-8") as f:
                f.write(self.latex.toPlainText())
            written += [base + "_weights.csv", base + "_equation.tex"]
        self.status.setText("Exported " + ", ".join(os.path.basename(p) for p in written) + "." + note)

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
            for j, k in enumerate(inputs, start=1):
                b = fits[k].get("bias")
                if b:
                    writer.writerow([f"b_{j}", self._weight_label(k), "", "bias volume [mm3]", " ".join(self._shells(k)),
                                     "", f"{b:.10g}", "", "", "", fit["error"], fit["tolerance"]])

    def _cancel_fit(self):
        """Stop a running equation fit; its result is ignored (see _refit)."""
        if self.fit_worker is not None:
            self.fit_worker.cancel()
            self._stopping.append(self.fit_worker)  # its thread ends at the next check; closing waits for it
            self.fit_worker = None
            self.regenerate.setEnabled(self.worker is None and bool(self.rows))

    def _busy(self):
        return [w for w in (self.worker, *self._stopping) if w is not None and w.isRunning()]

    def _stop_and_close(self):
        """Closing while a sweep or fit runs: cancel it and close once its thread has finished. Waiting
        for it here would freeze the window (Windows then reports the app as not responding)."""
        self._closing = True
        self._cancel_fit()
        if self.worker is not None:
            self.worker.cancel()
        self.status.setText("Stopping…")
        for w in self._busy():
            w.finished.connect(self._close_when_idle)
        self._close_when_idle()  # in case everything finished in the meantime

    def _close_when_idle(self):
        if not self._busy():
            QTimer.singleShot(0, self.reject)

    def reject(self):  # Esc, the Close button and the window's close button all end here
        if self._busy() or self.fit_worker is not None:
            self._stop_and_close()
            return
        super().reject()

    def closeEvent(self, event):
        if self._busy() or self.fit_worker is not None:
            event.ignore()
            self._stop_and_close()
            return
        super().closeEvent(event)
