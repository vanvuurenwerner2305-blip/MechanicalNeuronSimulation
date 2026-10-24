"""
Network tab: characterised neurons as blocks on a Simulink-style canvas, wired into a network that is evaluated
from the stored characterisations (app.network; nothing is simulated).

Library (left): characterised full neurons (*.mfn) and valve designs (*.mad) once added, the built-in blocks (sources
and probes, math, activation functions, logic and comparison, lookup) and the network's subsystems. Drag an entry
onto the canvas or double-click it; files can also be dropped on the canvas. Drag from an output port (right side of
a block) to an input port (left side) to wire them.
Subsystems: select blocks and press Ctrl+G to put them into a subsystem (its ports are made from the wires that
cross the selection); double-click a subsystem to open it, Backspace (or the location bar) to go back up. Inside, In
port and Out port blocks are its ports. Every instance of a subsystem is the same (editing one edits all); "Make
independent" gives one block its own copy. A subsystem can be exported as a library block (*.mnb) and imported into
other networks.
Block (right): the selected block's settings. Run (right): the network inputs' values, evaluation (automatic on
every change) and sweeps of one or two inputs, plotted below.
"""
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QMimeData, QPointF, QRectF, QSettings, Qt, QTimer, Signal
from qtpy.QtGui import (QBrush, QColor, QDrag, QFont, QFontMetricsF, QPainter, QPainterPath, QPainterPathStroker, QPen,
                        QPolygonF)
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                            QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene,
                            QGraphicsSimpleTextItem, QGraphicsView, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                            QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                            QPlainTextEdit, QProgressDialog, QPushButton, QScrollArea, QSpinBox, QStyle, QTableWidget,
                            QTableWidgetItem, QTabWidget, QToolBar, QToolButton, QTreeWidget, QTreeWidgetItem,
                            QDialog, QDialogButtonBox,
                            QVBoxLayout, QWidget)

try:
    from qtpy.QtWidgets import QAction
except ImportError:  # Qt6
    from qtpy.QtGui import QAction

from .network import (ACTIVATION, BLOCK_FILTER, BLOCK_SUFFIX, CLAMP, CONSTANT, CURRENT, EXTRAPOLATE, FILTER,
                      FUNCTION_TYPES, GROUPS, INPORT, INPUT, LOGIC, LOOKUP, MATH, MISSING, NEURON, OUTDATED, OUTPORT,
                      OUTPUT, ROLE_CONSTANT, ROLE_INPUT, ROLE_PARAMETER, ROLES, SOURCES, SUBSYSTEM, SUBSYSTEMS, SUFFIX, TYPE_NAMES, TYPES,
                      VALVE, Network, component_from_file, default_outputs, new_block, short_label, source_status, sweep_csv)

TITLE = "Network"
MIME = "application/x-mns-block"
GRID = 10.0
PORT_R = 5.0
ROW = 20.0
HEAD = 24.0
GROUP_COLORS = {MATH: "#607d8b", ACTIVATION: "#c2185b", LOGIC: "#00897b", LOOKUP: "#6d4c41"}
COLORS = {NEURON: "#7e57c2", VALVE: "#e0812a", INPUT: "#3a9a4a", CONSTANT: "#6aa84f", OUTPUT: "#3a73c4",
          INPORT: "#455a64", OUTPORT: "#455a64", SUBSYSTEM: "#283593"}
STATUS_COLORS = {OUTDATED: "#d07000", MISSING: "#c00000"}
MAX_LINES = 8            # a second swept input with at most this many values is drawn as lines, else colour maps

GUIDE = """
<h3>Network</h3>
<p>Build a network from characterised neurons without simulating anything: every block looks its response up
in the stored characterisation (multilinear interpolation).</p>
<ol>
<li><b>Add neuron…</b>: a full neuron (*.mfn) with a stored characterisation (Full neuron tab → Run
characterisation). <b>Add valve…</b>: a simulated pre-activation-to-activation design (*.mad). Files can also be
dropped on the canvas.</li>
<li>Drag a library entry onto the canvas (or double-click it): sources and probes (<b>Input</b> = a network
input, <b>Output</b> = a probe that is plotted), math, activation functions (sigmoid, tanh, ReLU, ...), logic and
comparison (true = 1, false = 0), lookup tables and subsystems.</li>
<li>Drag from an output port (right of a block) to an input port (left) to wire them. <b>Delete</b> removes the
selection. Wheel = zoom, middle button = pan, <b>F</b> = fit.</li>
<li><b>Subsystems</b>: select blocks and press <b>Ctrl+G</b>; the wires crossing the selection become its ports.
Double-click a subsystem to open it, <b>Backspace</b> to go up. Inside, <b>In port</b> / <b>Out port</b> blocks are
its ports. All instances of a subsystem share its contents; <i>Make independent</i> gives one its own copy.
Right-click it in the library to export it as a library block (*.mnb); <b>Import block…</b> adds one.</li>
<li>Select a neuron block to choose which swept parameters are <b>input ports</b> (an unconnected input is held
at its value) and which are <b>parameters</b>, and which recorded quantities are output ports. Outside the
characterised range a block extrapolates or clamps (red outline).</li>
<li><b>Run</b> tab: set the inputs (values appear on the ports at once), or sweep one or two inputs (F6).</li>
</ol>
<p>Connections are ideal: a block does not load the one that drives it. That holds for a valve's outputs; a
sealed chamber's pressure (dashed wire) wired into another chamber would in reality exchange volume with it.
Feedback loops are solved as algebraic loops; a sweep follows one branch of a bistable loop.</p>
"""


def snap(v):
    return round(v / GRID) * GRID


def _tidy(v):
    """A shown value: round-off around zero (e.g. -7.5e-13 at zero input) is shown as 0."""
    return 0.0 if abs(v) < 1e-9 else v


def block_color(kind):
    return COLORS.get(kind) or GROUP_COLORS.get(TYPES[kind].group, "#777777")


# -----------------------------
# Canvas items
# -----------------------------

class PortItem(QGraphicsEllipseItem):
    def __init__(self, block_item, name, output, label, unit):
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, block_item)
        self.block_item, self.name, self.output, self.label, self.unit = block_item, name, output, label, unit
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#333333"), 1.2))
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip(f"{'Output' if output else 'Input'} {label or name}" + (f" [{unit}]" if unit else "")
                        + ("\nDrag to an input port to connect" if output else "\nDrag to an output port to connect"))
        self.setZValue(2)
        self.wires = []

    def key(self):
        return self.block_item.block["id"], self.name

    def centre(self):
        return self.scenePos()

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor("#ffd54f")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor("#ffffff")))
        super().hoverLeaveEvent(event)


class BlockItem(QGraphicsRectItem):
    def __init__(self, window, block):
        super().__init__()
        self.window, self.block = window, block
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(1)
        self.flags_out = []
        self.value_text = ""
        self.inputs, self.outputs = {}, {}
        self.value_items = {}
        self.font = QFont()
        self.font.setPointSizeF(8.5)
        self.head_font = QFont()
        self.head_font.setPointSizeF(9)
        self.head_font.setBold(True)
        self.rebuild()
        self.setPos(QPointF(*block.get("pos", (0.0, 0.0))))

    @property
    def kind(self):
        return self.block["type"]

    def _labelled(self):
        return self.kind in (NEURON, VALVE, SUBSYSTEM)

    def _subtitle(self):
        return self.kind in (NEURON, VALVE, SUBSYSTEM)

    def rebuild(self):
        """Size and ports from the block (after its ports changed)."""
        for item in list(self.inputs.values()) + list(self.outputs.values()) + list(self.value_items.values()):
            item.setParentItem(None)
            if item.scene():
                item.scene().removeItem(item)
        net = self.window.network
        ins, outs = net.ports(self.block)
        fm = QFontMetricsF(self.font)
        in_labels = [net.port_label(self.block, p, output=False) for p in ins]
        out_labels = [net.port_label(self.block, p) for p in outs]
        width = 1.1 * (max([fm.horizontalAdvance(t) for t in in_labels] + [0]) +
                       max([fm.horizontalAdvance(t) for t in out_labels] + [0])) + 44
        symbol = TYPES[self.kind].symbol(self.block) if self.kind in FUNCTION_TYPES else ""
        bold = QFont(self.font)
        bold.setPointSizeF(10)
        bold.setBold(True)
        width = max(width, QFontMetricsF(self.head_font).horizontalAdvance(self.block["name"]) + 24,
                    QFontMetricsF(bold).horizontalAdvance(symbol) + 40, 160 if self._labelled() else 110)
        rows = max(len(ins), len(outs), 1)
        extra = 14 if self._subtitle() else 0
        curve = 30 if TYPES[self.kind].curve else 0
        height = HEAD + rows * ROW + 8 + extra + curve
        self.setRect(0, 0, snap(width + GRID / 2), snap(height + GRID / 2))
        self.inputs, self.outputs, self.value_items = {}, {}, {}
        top = HEAD + 4 + extra
        h = self.rect().height() - top - 4
        for k, p in enumerate(ins):
            item = PortItem(self, p, False, in_labels[k], net.unit(self.block, p, output=False))
            item.setPos(0, top + (k + 0.5) * (h / max(len(ins), 1)) if not self._labelled() else
                        top + (k + 0.5) * ROW)
            self.inputs[p] = item
        for k, p in enumerate(outs):
            item = PortItem(self, p, True, out_labels[k], net.unit(self.block, p))
            y = top + (k + 0.5) * (h / max(len(outs), 1)) if not self._labelled() else top + (k + 0.5) * ROW
            item.setPos(self.rect().width(), y)
            self.outputs[p] = item
            text = QGraphicsSimpleTextItem("", self)
            text.setFont(self.font)
            text.setBrush(QBrush(QColor("#1a4f9c")))
            text.setPos(self.rect().width() + 8, y - 15)
            self.value_items[p] = text
        self.update()

    def port(self, name, output):
        return (self.outputs if output else self.inputs).get(name)

    def set_values(self, values, prefix, flags, units):
        """Show the evaluated output values next to the ports, the value inside Input/Constant/Output blocks."""
        self.flags_out = flags or []
        fid = prefix + self.block["id"]
        for p, item in self.value_items.items():
            v = values.get((fid, p))
            item.setText("" if v is None else f"{_tidy(v):.4g} {units.get(p, '')}".strip())
        if self.kind == OUTPUT:
            v = values.get((fid, "in"))
            self.value_text = "-" if v is None or not math.isfinite(v) else f"{_tidy(v):.5g}"
        elif self.kind in (INPUT, CONSTANT):
            v = values.get((fid, "out"))
            self.value_text = "" if v is None else f"{_tidy(v):.5g} {self.block.get('unit', '')}".strip()
        elif self.kind in (INPORT, OUTPORT):      # the value passing through the subsystem's port
            v = values.get((fid, "out"))
            self.value_text = "" if v is None or not math.isfinite(v) else f"{_tidy(v):.5g}"
        if self.flags_out:
            errors = [f for f in self.flags_out if f.startswith("error")]
            self.setToolTip("; ".join(errors) if errors else "Outside the characterised range: " +
                            ", ".join(self.flags_out) + (" (clamped)" if self.block.get("range") == CLAMP
                                                         else " (extrapolated)"))
        else:
            self.setToolTip(TYPES[self.kind].tip)
        self.update()

    def paint(self, painter, option, widget=None):
        r = self.rect()
        kind = self.kind
        colour = QColor(block_color(kind))
        painter.setRenderHint(QPainter.Antialiasing)
        border = QColor("#d32f2f") if self.flags_out else (QColor("#1e88e5") if self.isSelected() else QColor("#444444"))
        painter.setPen(QPen(border, 2.5 if (self.flags_out or self.isSelected()) else 1.2))
        painter.setBrush(QBrush(QColor("#fbfbfb")))
        if kind == SUBSYSTEM:      # a stacked look: a block made of blocks
            painter.drawRoundedRect(r.translated(4, 4), 6, 6)
        painter.drawRoundedRect(r, 6, 6)
        head = QPainterPath()
        head.setFillRule(Qt.WindingFill)        # the union of the rounded and the square-cornered parts
        head.addRoundedRect(QRectF(r.x(), r.y(), r.width(), HEAD), 6, 6)
        head.addRect(QRectF(r.x(), r.y() + HEAD / 2, r.width(), HEAD / 2))
        painter.fillPath(head.simplified(), QBrush(colour))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(self.head_font)
        title = self.block["name"]
        if kind in (INPORT, OUTPORT):
            title = f"{self.block.get('index', '')}: {title}"
        painter.drawText(QRectF(r.x() + 6, r.y(), r.width() - 12, HEAD), Qt.AlignVCenter | Qt.AlignLeft, title)
        painter.setFont(self.font)
        painter.setPen(QPen(QColor("#555555")))
        net = self.window.network
        if self._subtitle():
            if kind == SUBSYSTEM:
                d = net.definition_of(self.block)
                n = len(net.instances(self.block.get("definition"))) if d else 0
                sub = (d["name"] if d else "(missing)") + (f"  ({n} instances)" if n > 1 else "") + "  ▸ double-click"
            else:
                component = net.component_of(self.block)
                # held variables; trainable parameters marked with a diamond
                params = [f"{'◆' if s['role'] == ROLE_PARAMETER else ''}{n.rsplit('.', 1)[-1]}={s['value']:.3g}"
                          for n, s in self.block["axes"].items()
                          if s["role"] != ROLE_INPUT and len(self._axis_values(n)) > 1]
                sub = (component["name"] if component else "(missing component)") + \
                    ("  " + ", ".join(params) if params else "")
            painter.drawText(QRectF(r.x() + 6, r.y() + HEAD + 1, r.width() - 12, 14), Qt.AlignLeft | Qt.AlignVCenter,
                             QFontMetricsF(self.font).elidedText(sub, Qt.ElideRight, r.width() - 12))
        painter.setPen(QPen(QColor("#222222")))
        for item in self.inputs.values():
            if item.label:
                painter.drawText(QRectF(10, item.y() - ROW / 2, r.width() - 20, ROW), Qt.AlignVCenter | Qt.AlignLeft,
                                 item.label)
        for item in self.outputs.values():
            if item.label:
                painter.drawText(QRectF(10, item.y() - ROW / 2, r.width() - 20, ROW), Qt.AlignVCenter | Qt.AlignRight,
                                 item.label)
        symbol = TYPES[kind].symbol(self.block) if kind in FUNCTION_TYPES else ""
        if symbol and self.block.get("trainable"):
            symbol = "◆ " + symbol
        if kind in (INPUT, CONSTANT, OUTPUT, INPORT, OUTPORT):
            symbol = self.value_text
        body = QRectF(r.x() + 12, r.y() + HEAD, r.width() - 24, r.height() - HEAD)
        if TYPES[kind].curve:
            self._draw_curve(painter, QRectF(body.x(), body.y() + 4, body.width(), body.height() - 22))
            body = QRectF(body.x(), r.bottom() - 20, body.width(), 18)
        if symbol:
            f = QFont(self.font)
            f.setPointSizeF(10 if not TYPES[kind].curve else 8.5)
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QPen(QColor("#1a4f9c") if kind == OUTPUT else QColor("#222222")))
            painter.drawText(body, Qt.AlignCenter, QFontMetricsF(f).elidedText(symbol, Qt.ElideRight, body.width()))

    def _draw_curve(self, painter, box):
        """A small plot of a one-input function block over its typical range."""
        t = TYPES[self.kind]
        try:
            lo, hi = t.curve(self.block)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                return
            xs = np.linspace(lo, hi, 60)
            ys = np.array([t.fn(self.block, [x]) for x in xs], float)
        except Exception:
            return
        ok = np.isfinite(ys)
        if ok.sum() < 2:
            return
        y0, y1 = ys[ok].min(), ys[ok].max()
        if y1 - y0 < 1e-12:
            y0, y1 = y0 - 1, y1 + 1
        painter.setPen(QPen(QColor("#dddddd"), 0.8))
        painter.drawLine(QPointF(box.left(), box.bottom()), QPointF(box.right(), box.bottom()))
        points = QPolygonF([QPointF(box.left() + (x - lo) / (hi - lo) * box.width(),
                                    box.bottom() - (y - y0) / (y1 - y0) * box.height())
                            for x, y in zip(xs[ok], ys[ok])])
        painter.setPen(QPen(QColor(block_color(self.kind)), 1.6))
        painter.drawPolyline(points)

    def _axis_values(self, name):
        component = self.window.network.component_of(self.block)
        if not component:
            return []
        return next((a["values"] for a in component["axes"] if a["name"] == name), [])

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            return QPointF(snap(value.x()), snap(value.y()))
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.block["pos"] = [self.pos().x(), self.pos().y()]
            for item in list(self.inputs.values()) + list(self.outputs.values()):
                for wire in item.wires:
                    wire.update_path()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        if not self.isSelected():
            self.scene().clearSelection()
            self.setSelected(True)
        self.window.block_menu(self.block, event.screenPos())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.kind == SUBSYSTEM:
            QTimer.singleShot(0, lambda: self.window.enter(self.block))
        else:
            self.window.show_block_tab()
        super().mouseDoubleClickEvent(event)


def wire_path(a, b, below=None):
    """From output port a to input port b; a wire that runs backwards (feedback) goes round under the blocks
    (at y = below) when that is given."""
    if below is not None and b.x() < a.x() + 30:
        r = 12.0
        path = QPainterPath(a)
        path.lineTo(a.x() + 20 - r, a.y())
        path.quadTo(QPointF(a.x() + 20, a.y()), QPointF(a.x() + 20, a.y() + r))
        path.lineTo(a.x() + 20, below - r)
        path.quadTo(QPointF(a.x() + 20, below), QPointF(a.x() + 20 - r, below))
        path.lineTo(b.x() - 20 + r, below)
        path.quadTo(QPointF(b.x() - 20, below), QPointF(b.x() - 20, below - r))
        path.lineTo(b.x() - 20, b.y() + r)
        path.quadTo(QPointF(b.x() - 20, b.y()), QPointF(b.x() - 20 + r, b.y()))
        path.lineTo(b)
        return path
    dx = max(40.0, abs(b.x() - a.x()) / 2)
    path = QPainterPath(a)
    path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
    return path


class WireItem(QGraphicsPathItem):
    def __init__(self, wire, src, dst, ideal):
        super().__init__()
        self.wire, self.src, self.dst = wire, src, dst
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setZValue(0)
        self.ideal = ideal
        if not ideal:
            self.setToolTip("A sealed chamber's quantity: wired as an ideal signal (the receiving chamber's volume "
                            "change does not act back on it)")
        src.wires.append(self)
        dst.wires.append(self)
        self.update_path()

    def update_path(self):
        below = max(self.src.block_item.sceneBoundingRect().bottom(),
                    self.dst.block_item.sceneBoundingRect().bottom()) + 20
        self.setPath(wire_path(self.src.centre(), self.dst.centre(), below))

    def paint(self, painter, option, widget=None):
        pen = QPen(QColor("#1e88e5") if self.isSelected() else QColor("#333333"), 2.4 if self.isSelected() else 1.6)
        if not self.ideal:
            pen.setStyle(Qt.DashLine)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pen)
        painter.drawPath(self.path())

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(8)
        return stroker.createStroke(self.path())

    def detach(self):
        for port in (self.src, self.dst):
            if self in port.wires:
                port.wires.remove(self)


class NetworkScene(QGraphicsScene):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.drag_from = None
        self.drag_path = None
        self.setSceneRect(-5000, -5000, 10000, 10000)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#ffffff" if not self.window.path else "#fbfbff"))
        step = GRID * 2
        painter.setPen(QPen(QColor("#e6e6e6"), 0))
        x = math.floor(rect.left() / step) * step
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = math.floor(rect.top() / step) * step
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

    def _port_at(self, pos):
        for item in self.items(pos):
            if isinstance(item, PortItem):
                return item
        return None

    def mousePressEvent(self, event):
        port = self._port_at(event.scenePos()) if event.button() == Qt.LeftButton else None
        if port is not None:
            self.drag_from = port
            self.drag_path = QGraphicsPathItem()
            self.drag_path.setPen(QPen(QColor("#1e88e5"), 1.6, Qt.DashLine))
            self.drag_path.setZValue(3)
            self.addItem(self.drag_path)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_from is not None:
            a, b = self.drag_from.centre(), event.scenePos()
            self.drag_path.setPath(wire_path(a, b) if self.drag_from.output else wire_path(b, a))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_from is not None:
            start, self.drag_from = self.drag_from, None
            self.removeItem(self.drag_path)
            self.drag_path = None
            end = self._port_at(event.scenePos())
            if end is not None and end.output != start.output:
                src, dst = (start, end) if start.output else (end, start)
                self.window.connect_ports(src.key(), dst.key())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class NetworkView(QGraphicsView):
    def __init__(self, window):
        super().__init__(window.scene)
        self.window = window
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setAcceptDrops(True)
        self._pan = None

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if 0.2 <= self.transform().m11() * factor <= 4.0:
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            d = event.pos() - self._pan
            self._pan = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._pan is not None:
            self._pan = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _accepts(self, event):
        md = event.mimeData()
        return md.hasFormat(MIME) or (md.hasUrls() and any(
            u.toLocalFile().lower().endswith((".mfn", ".mad", BLOCK_SUFFIX)) for u in md.urls()))

    def dragEnterEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()
        pos = self.mapToScene(event.pos())
        if md.hasFormat(MIME):
            kind, _, ref = bytes(md.data(MIME)).decode("utf-8").partition("|")
            self.window.add_block(kind, ref or None, pos)
            event.acceptProposedAction()
        elif md.hasUrls():
            for k, url in enumerate(md.urls()):
                path = url.toLocalFile()
                at = pos + QPointF(0, 120 * k)
                if path.lower().endswith(BLOCK_SUFFIX):
                    did = self.window.import_block(path)
                    if did:
                        self.window.add_block(SUBSYSTEM, did, at)
                elif path.lower().endswith((".mfn", ".mad")):
                    cid = self.window.add_component(path)
                    if cid:
                        self.window.add_block(self.window.network.components[cid]["kind"], cid, at)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace and self.window.path:
            self.window.go_up()
            event.accept()
            return
        super().keyPressEvent(event)


class LibraryTree(QTreeWidget):
    """The block library; its entries are dragged onto the canvas."""

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)

    def startDrag(self, actions):
        item = self.currentItem()
        data = item.data(0, Qt.UserRole) if item else None
        if not data:
            return
        md = QMimeData()
        md.setData(MIME, data.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.exec_(Qt.CopyAction)


# -----------------------------
# Analysing one block
# -----------------------------

def _default_points(n):
    """Grid points per swept variable, so that a block evaluates in seconds."""
    return {0: 1, 1: 81, 2: 41, 3: 15, 4: 9}.get(n, 6)


class AnalyseDialog(QDialog):
    """Which of a block's inputs and trainable parameters to sweep, over what range, before it is shown in the
    Analysis tab; the rest are held (inputs at their present value in the network)."""

    def __init__(self, parent, network, block, held, prefix=""):
        super().__init__(parent)
        self.setWindowTitle(f"Analyse {block['name']}")
        self.network, self.block = network, block
        self.rows = []
        ins, _ = network.ports(block)
        for port in ins:
            r = network.port_range(block, port) or (-20.0, 20.0)
            self.rows.append({"kind": "input", "port": port, "label": network.port_label(block, port, output=False)
                              or (port if len(ins) > 1 else "u"), "unit": network.unit(block, port, output=False),
                              "range": r, "held": float(held.get(port, 0.0)), "sweep": True})
        for prm in network.block_parameters(block):
            if prm["min"] is not None:
                r = (prm["min"], prm["max"])
            else:
                v = prm["value"]
                r = (v - max(abs(v), 1.0) * 0.5, v + max(abs(v), 1.0) * 0.5)
            self.rows.append({"kind": "parameter", "parameter": prm, "label": prm["name"], "unit": prm["unit"],
                              "range": r, "held": prm["value"], "sweep": False})
        seen = {}
        for row in self.rows:             # unique labels (they name the axes)
            n = seen.get(row["label"], 0)
            seen[row["label"]] = n + 1
            if n:
                row["label"] = f"{row['label']} ({n + 1})"
        layout = QVBoxLayout(self)
        note = QLabel("The block is evaluated on its own on every combination of the ticked variables (looked up "
                      "in its characterisations, nothing is simulated) and shown in the Analysis tab. Unticked "
                      "inputs are held at their present value in the network, unticked parameters at theirs.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(len(self.rows), 6)
        self.table.setHorizontalHeaderLabels(["Sweep", "Variable", "From", "To", "Points", "Else held at"])
        self.table.verticalHeader().setVisible(False)
        self.widgets = []
        for r, row in enumerate(self.rows):
            check = QCheckBox()
            check.setChecked(row["sweep"])
            check.toggled.connect(lambda _=False: self._changed(True))
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(6, 0, 0, 0)
            h.addWidget(check)
            self.table.setCellWidget(r, 0, holder)
            cell = QTableWidgetItem(("◆ " if row["kind"] == "parameter" else "") + row["label"]
                                    + (f" [{row['unit']}]" if row["unit"] else ""))
            cell.setFlags(Qt.ItemIsEnabled)
            cell.setToolTip("A trainable parameter" if row["kind"] == "parameter" else "An input port")
            self.table.setItem(r, 1, cell)
            spins = []
            for col, value in ((2, row["range"][0]), (3, row["range"][1]), (5, row["held"])):
                spin = QDoubleSpinBox()
                spin.setRange(-1e9, 1e9)
                spin.setDecimals(4)
                spin.setValue(float(value))
                self.table.setCellWidget(r, col, spin)
                spins.append(spin)
            points = QSpinBox()
            points.setRange(2, 401)
            points.valueChanged.connect(lambda _=0: self._changed(False))
            self.table.setCellWidget(r, 4, points)
            self.widgets.append({"check": check, "from": spins[0], "to": spins[1], "held": spins[2], "points": points})
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (0, 2, 3, 4, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setMinimumWidth(720)
        self.table.setMinimumHeight(min(80 + 32 * len(self.rows), 420))
        layout.addWidget(self.table)
        self.total = QLabel()
        layout.addWidget(self.total)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Analyse")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._changed(True)

    def _changed(self, reset_points):
        swept = [w for w in self.widgets if w["check"].isChecked()]
        for w in self.widgets:
            on = w["check"].isChecked()
            for key in ("from", "to", "points"):
                w[key].setEnabled(on)
            w["held"].setEnabled(not on)
            if reset_points:
                w["points"].blockSignals(True)
                w["points"].setValue(_default_points(len(swept)))
                w["points"].blockSignals(False)
        n = int(np.prod([w["points"].value() for w in swept])) if swept else 1
        self.total.setText(f"{n} evaluation(s)" + (" - this may take a while" if n > 50000 else ""))

    def sweeps(self):
        """(sweeps, held inputs {port: value}, held parameters [(parameter, value)])."""
        sweeps, held, params = [], {}, []
        for row, w in zip(self.rows, self.widgets):
            if w["check"].isChecked():
                s = {"kind": row["kind"], "label": row["label"], "unit": row["unit"], "present": w["held"].value(),
                     "values": np.linspace(w["from"].value(), w["to"].value(), w["points"].value()).tolist()}
                s["port" if row["kind"] == "input" else "parameter"] = row.get("port", row.get("parameter"))
                sweeps.append(s)
            elif row["kind"] == "input":
                held[row["port"]] = w["held"].value()
            else:
                params.append((row["parameter"], w["held"].value()))
        return sweeps, held, params


# -----------------------------
# Window
# -----------------------------

class NetworkWindow(QMainWindow):
    TITLE = TITLE
    open_source = Signal(str)
    analyse = Signal(str, object)      # a neuron's .mfn and {"axes": the block's variables, "key": its first output}
    analyse_data = Signal(object, object)   # a block's computed response (characterisation format) and its settings

    def __init__(self):
        super().__init__()
        self.worker = None                 # (AppWindow.closeEvent)
        self.network = Network()
        self.path = []                     # subsystem instance ids from the top level to the one shown
        self.items = {}                    # block id -> BlockItem (of the shown container)
        self.wire_items = []
        self.result = None
        self.sweep_result = None
        self.statuses = {}
        self.dirty = False
        self._building = False
        self.scene = NetworkScene(self)
        self.scene.selectionChanged.connect(self._selection_changed)
        self._build_ui()
        self._build_actions()
        self._fill_library()
        self._fill_run()
        self._show_location()
        self._show_block()
        self._update_title()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_sources)
        self.timer.start(2000)
        self.log_message("Network: Add neuron… (a characterised *.mfn) or Add valve… (*.mad), drag blocks onto the "
                         "canvas and wire output ports to input ports. Help → Quick guide explains the rest.")

    # -----------------------------
    # Where we are (the top level or inside a subsystem)
    # -----------------------------

    def container(self):
        c = self.network.root
        for bid in self.path:
            block = self.network.block(bid, c)
            c = self.network.definition_of(block) if block else None
            if c is None:
                self.path = []
                return self.network.root
        return c

    def prefix(self):
        return "".join(bid + "/" for bid in self.path)

    def enter(self, block):
        if block["type"] != SUBSYSTEM or self.network.definition_of(block) is None:
            return
        self.path.append(block["id"])
        self._after_move()

    def go_up(self, levels=1):
        if not self.path:
            return
        self.path = self.path[:-levels] if levels <= len(self.path) else []
        self._after_move()

    def _after_move(self):
        self._show_location()
        self._fill_library()
        self._rebuild_scene()
        QTimer.singleShot(0, self.fit_view)

    def _show_location(self):
        bar = self.location_bar
        bar.clear()
        names, c = [], self.network.root
        for bid in self.path:
            block = self.network.block(bid, c)
            names.append(block["name"] if block else bid)
            c = self.network.definition_of(block) if block else c
        crumbs = ["Network"] + names
        for k, name in enumerate(crumbs):
            if k:
                bar.addWidget(QLabel(" ▸ "))
            button = QToolButton()
            button.setText(name)
            button.setAutoRaise(True)
            if k == len(crumbs) - 1:
                f = button.font()
                f.setBold(True)
                button.setFont(f)
            button.clicked.connect(lambda _=False, up=len(crumbs) - 1 - k: up and self.go_up(up))
            bar.addWidget(button)
        if self.path:
            d = self.container()
            did = next((i for i, x in self.network.definitions.items() if x is d), None)
            n = len(self.network.instances(did))
            note = QLabel(f"   (subsystem “{d['name']}”" + (f", {n} instances share these contents" if n > 1 else "")
                          + "; Backspace = up)")
            note.setStyleSheet("color: #555;")
            bar.addWidget(note)
        self.a_up.setEnabled(bool(self.path))

    # -----------------------------
    # UI
    # -----------------------------

    def _build_ui(self):
        self.view = NetworkView(self)
        self.setCentralWidget(self.view)

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self.library = LibraryTree()
        self.library.itemDoubleClicked.connect(self._library_double_clicked)
        self.library.setContextMenuPolicy(Qt.CustomContextMenu)
        self.library.customContextMenuRequested.connect(self._library_menu)
        v.addWidget(self.library)
        row = QHBoxLayout()
        for text, slot in (("Add neuron…", lambda: self.add_component_dialog(NEURON)),
                           ("Add valve…", lambda: self.add_component_dialog(VALVE)),
                           ("Import block…", self.import_block_dialog)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        hint = QLabel("Drag onto the canvas or double-click. Right-click a component or subsystem for more.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        v.addWidget(hint)
        dock = QDockWidget("Library", self)
        dock.setObjectName("network_library_dock")
        dock.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.tabs = QTabWidget()
        self.block_area = QScrollArea()
        self.block_area.setWidgetResizable(True)
        self.tabs.addTab(self.block_area, "Block")
        self.tabs.addTab(self._make_run_tab(), "Run")
        dock = QDockWidget("Network", self)
        dock.setObjectName("network_controls_dock")
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(470)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.figure = Figure(figsize=(7, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.bottom = QTabWidget()
        self.bottom.addTab(self.canvas, "Plot")
        self.bottom.addTab(self.log, "Messages")
        dock = QDockWidget("Results", self)
        dock.setObjectName("network_results_dock")
        dock.setWidget(self.bottom)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [260], Qt.Vertical)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)

    def _make_run_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.auto = QCheckBox("Evaluate on every change")
        self.auto.setChecked(True)
        self.auto.toggled.connect(lambda on: on and self.evaluate())
        layout.addWidget(self.auto)
        box = QGroupBox("Network inputs")
        self.inputs_form = QFormLayout(box)
        layout.addWidget(box)
        b = QPushButton("Evaluate (F5)")
        b.clicked.connect(self.evaluate)
        layout.addWidget(b)
        box = QGroupBox("Trainable parameters")
        self.params_form = QFormLayout(box)
        box.setToolTip("Every variable or setting set to Parameter: what training will change. Neuron variables stay "
                       "within their characterised range.")
        layout.addWidget(box)
        box = QGroupBox("Outputs")
        v = QVBoxLayout(box)
        self.probe_label = QLabel("-")
        self.probe_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.probe_label.setWordWrap(True)
        v.addWidget(self.probe_label)
        layout.addWidget(box)

        box = QGroupBox("Sweep")
        form = QFormLayout(box)
        self.x_combo = QComboBox()
        self.x_combo.currentIndexChanged.connect(lambda _: self._sweep_x_changed())
        form.addRow("x (input)", self.x_combo)
        self.x_from, self.x_to, self.x_points = self._range_row(form, "x from / to / points", self._sweep_x_edited)
        self.y_combo = QComboBox()
        self.y_combo.currentIndexChanged.connect(lambda _: self._sweep_y_changed())
        form.addRow("y (input)", self.y_combo)
        self.y_from, self.y_to, self.y_points = self._range_row(form, "y from / to / points", self._sweep_y_edited)
        row = QHBoxLayout()
        b = QPushButton("Run sweep (F6)")
        b.clicked.connect(self.run_sweep)
        row.addWidget(b)
        b = QPushButton("Export CSV…")
        b.clicked.connect(self.export_csv)
        row.addWidget(b)
        form.addRow(row)
        note = QLabel(f"With a y input of up to {MAX_LINES} values every Output gets a panel with one curve per y "
                      "value, otherwise a colour map. Red points lie outside a block's characterised range, black "
                      "crosses are loops that did not converge.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        form.addRow(note)
        layout.addWidget(box)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        return scroll

    def _range_row(self, form, label, slot):
        row = QHBoxLayout()
        a, b = QDoubleSpinBox(), QDoubleSpinBox()
        for s in (a, b):
            s.setRange(-1e6, 1e6)
            s.setDecimals(4)
            s.valueChanged.connect(lambda _: slot())
            row.addWidget(s)
        n = QSpinBox()
        n.setRange(2, 2001)
        n.valueChanged.connect(lambda _: slot())
        row.addWidget(n)
        form.addRow(label, row)
        return a, b, n

    def _action(self, text, slot, shortcut=None, icon=None, tip=None, view_only=False):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        action.triggered.connect(slot)
        if view_only:          # keys that would clash with editing text elsewhere act on the canvas only
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            self.view.addAction(action)
        return action

    def _build_actions(self):
        S = QStyle
        self.a_new = self._action("New network", self.new_network, "Ctrl+N", S.SP_FileIcon)
        self.a_open = self._action("Open network…", self.open_project, "Ctrl+O", S.SP_DialogOpenButton,
                                   f"Open a neuron network (*{SUFFIX})")
        self.a_save = self._action("Save", lambda: self.save_project(False), "Ctrl+S", S.SP_DialogSaveButton)
        self.a_save_as = self._action("Save as…", lambda: self.save_project(True), "Ctrl+Shift+S")
        self.a_add_neuron = self._action("Add neuron…", lambda: self.add_component_dialog(NEURON), "Ctrl+I",
                                         S.SP_FileDialogNewFolder, "Add a characterised full neuron (*.mfn) to the "
                                                                   "library")
        self.a_add_valve = self._action("Add valve…", lambda: self.add_component_dialog(VALVE), "Ctrl+Shift+I",
                                        None, "Add a simulated valve design (*.mad) to the library")
        self.a_import_block = self._action("Import library block…", self.import_block_dialog, None, None,
                                           f"Add a subsystem saved as a library block (*{BLOCK_SUFFIX})")
        self.a_export = self._action("Export sweep CSV…", self.export_csv)
        self.a_delete = self._action("Delete", self.delete_selected, "Delete", S.SP_TrashIcon,
                                     "Delete the selected blocks and wires", view_only=True)
        self.a_select_all = self._action("Select all", self._select_all, "Ctrl+A", view_only=True)
        self.a_group = self._action("Create subsystem from selection", lambda: self.group_selected(), "Ctrl+G",
                                    S.SP_DirIcon, "Put the selected blocks into a new subsystem", view_only=True)
        self.a_open_sub = self._action("Open subsystem", self._open_selected, None, None,
                                       "Show the selected subsystem's contents (or double-click it)")
        self.a_up = self._action("Up", lambda: self.go_up(), "Alt+Up", S.SP_ArrowUp, "Back to the level above")
        self.a_eval = self._action("Evaluate", self.evaluate, "F5", S.SP_MediaPlay, "Evaluate the network at the "
                                                                                    "inputs' values")
        self.a_sweep = self._action("Run sweep", self.run_sweep, "F6", S.SP_BrowserReload,
                                    "Sweep one or two inputs and plot the Output blocks")
        self.a_fit = self._action("Fit to window", self.fit_view, "F", view_only=True)
        self.a_guide = self._action("Quick guide", self.show_guide, "F1")

        menu = self.menuBar()
        m = menu.addMenu("&File")
        for a in (self.a_new, self.a_open, self.a_save, self.a_save_as, None, self.a_add_neuron, self.a_add_valve,
                  self.a_import_block, None, self.a_export):
            m.addSeparator() if a is None else m.addAction(a)
        m = menu.addMenu("&Edit")
        for a in (self.a_delete, self.a_select_all, None, self.a_group, self.a_open_sub, self.a_up):
            m.addSeparator() if a is None else m.addAction(a)
        m = menu.addMenu("&View")
        m.addAction(self.a_fit)
        m = menu.addMenu("&Simulation")
        m.addAction(self.a_eval)
        m.addAction(self.a_sweep)
        m = menu.addMenu("&Help")
        m.addAction(self.a_guide)

        tb = QToolBar("Main")
        tb.setObjectName("network_toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_open, self.a_save, None, self.a_add_neuron, self.a_add_valve, None, self.a_eval,
                  self.a_sweep, None, self.a_group, self.a_up, self.a_delete, self.a_fit):
            tb.addSeparator() if a is None else tb.addAction(a)
        self.addToolBar(tb)
        self.addToolBarBreak()
        self.location_bar = QToolBar("Location")
        self.location_bar.setObjectName("network_location")
        self.location_bar.setMovable(False)
        self.addToolBar(self.location_bar)

    def log_message(self, text):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def show_guide(self):
        QMessageBox.information(self, "Quick guide - network", GUIDE)

    def _update_title(self):
        name = Path(self.network.path).name if self.network.path else "untitled"
        self.setWindowTitle(f"{TITLE} - {name}{' *' if self.dirty else ''}")

    def _set_dirty(self):
        self.dirty = True
        self._update_title()

    # -----------------------------
    # Library
    # -----------------------------

    def _fill_library(self):
        expanded = {self.library.topLevelItem(i).text(0): self.library.topLevelItem(i).isExpanded()
                    for i in range(self.library.topLevelItemCount())}
        self.library.clear()
        groups = {}
        for key, title in ((NEURON, "Characterised neurons"), (VALVE, "Valves (pre-activation → activation)")) + \
                tuple((g, g) for g in GROUPS):
            item = QTreeWidgetItem([title])
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            item.setFlags(Qt.ItemIsEnabled)
            self.library.addTopLevelItem(item)
            groups[key] = item
        for cid, c in self.network.components.items():
            status = self.statuses.get(cid, CURRENT)
            n = sum(1 for x in c["converged"] if x is not None)
            axes = ", ".join(f"{short_label(a['name'])} ({len(a['values'])})" for a in c["axes"])
            item = QTreeWidgetItem([c["name"] + (f"  — {status}" if status != CURRENT else "")])
            item.setData(0, Qt.UserRole, f"{c['kind']}|{cid}")
            item.setToolTip(0, f"{c.get('source') or '(no source)'}\n{n} point(s); swept: {axes}\nrecorded: "
                               + ", ".join(k["key"] for k in c["keys"]))
            if status in STATUS_COLORS:
                item.setForeground(0, QBrush(QColor(STATUS_COLORS[status])))
                item.setIcon(0, self.style().standardIcon(QStyle.SP_MessageBoxWarning))
            groups[c["kind"]].addChild(item)
        inside = bool(self.path)
        for kind, t in TYPES.items():
            if kind in (NEURON, VALVE, SUBSYSTEM) or (kind in (INPORT, OUTPORT) and not inside):
                continue
            item = QTreeWidgetItem([t.title])
            item.setData(0, Qt.UserRole, f"{kind}|")
            if t.tip:
                item.setToolTip(0, t.tip)
            groups[t.group].addChild(item)
        item = QTreeWidgetItem(["New empty subsystem"])
        item.setData(0, Qt.UserRole, f"{SUBSYSTEM}|")
        item.setToolTip(0, "Or select blocks on the canvas and press Ctrl+G")
        groups[SUBSYSTEMS].addChild(item)
        for did, d in self.network.definitions.items():
            n = len(self.network.instances(did))
            item = QTreeWidgetItem([f"▣ {d['name']}" + (f"  ({n})" if n else "  (unused)")])
            item.setData(0, Qt.UserRole, f"{SUBSYSTEM}|{did}")
            ins = [b["name"] for b in d["blocks"] if b["type"] == INPORT]
            outs = [b["name"] for b in d["blocks"] if b["type"] == OUTPORT]
            item.setToolTip(0, f"{len(d['blocks'])} block(s); in: {', '.join(ins) or '-'}; out: {', '.join(outs) or '-'}"
                               "\nRight-click: open, rename, duplicate, export, delete")
            groups[SUBSYSTEMS].addChild(item)
        if not groups[NEURON].childCount():
            hint = QTreeWidgetItem(["(Add neuron… to add a characterised *.mfn)"])
            hint.setFlags(Qt.ItemIsEnabled)
            hint.setForeground(0, QBrush(QColor("#888888")))
            groups[NEURON].addChild(hint)
        for title, item in ((self.library.topLevelItem(i).text(0), self.library.topLevelItem(i))
                            for i in range(self.library.topLevelItemCount())):
            item.setExpanded(expanded.get(title, title not in (LOOKUP,)))

    def _library_double_clicked(self, item, _column=0):
        data = item.data(0, Qt.UserRole)
        if data:
            kind, _, ref = data.partition("|")
            centre = self.view.mapToScene(self.view.viewport().rect().center())
            self.add_block(kind, ref or None, centre + QPointF(20 * (len(self.container()["blocks"]) % 5), 0))

    def _library_menu(self, pos):
        item = self.library.itemAt(pos)
        data = item.data(0, Qt.UserRole) if item else None
        if data and data.partition("|")[0] in FUNCTION_TYPES:
            kind = data.partition("|")[0]
            menu = QMenu(self)
            response = menu.addAction(f"Analyse a {TYPE_NAMES[kind]} block…")
            if menu.exec_(self.library.viewport().mapToGlobal(pos)) is response:
                block = new_block(kind)
                block.update(id="analysed", name=TYPE_NAMES[kind])
                QTimer.singleShot(0, lambda: self.analyse_response(block, in_network=False))
            return
        if not data or not data.partition("|")[2]:
            return
        kind, _, ref = data.partition("|")
        if kind == SUBSYSTEM:
            return self._definition_menu(ref, self.library.viewport().mapToGlobal(pos))
        c = self.network.components[ref]
        menu = QMenu(self)
        update = menu.addAction("Update from source")
        update.setEnabled(bool(c.get("source")) and Path(c["source"]).exists())
        open_src = menu.addAction(f"Open {Path(c['source']).name} in its tab" if c.get("source") else "Open source")
        open_src.setEnabled(bool(c.get("source")) and Path(c["source"]).exists())
        analyse = None
        response = menu.addAction("Analyse its response…")
        if c["kind"] == NEURON:
            analyse = menu.addAction("Open its characterisation in Analysis…")
            analyse.setEnabled(bool(c.get("source")) and Path(c["source"]).exists())
        remove = menu.addAction("Remove from the library")
        chosen = menu.exec_(self.library.viewport().mapToGlobal(pos))
        if chosen is update:
            self.update_component(ref)
        elif chosen is not None and chosen is analyse:
            self.analyse.emit(c["source"], {"axes": {}, "key": default_outputs(c)[0], "block": c["name"]})
        elif chosen is not None and chosen is response:
            block = new_block(c["kind"], c)
            block.update(id="analysed", name=c["name"])
            QTimer.singleShot(0, lambda: self.analyse_response(block, in_network=False))
        elif chosen is open_src:
            self.open_source.emit(c["source"])
        elif chosen is remove:
            try:
                self.network.remove_component(ref)
            except ValueError as exc:
                QMessageBox.information(self, TITLE, str(exc))
                return
            self._set_dirty()
            self._fill_library()

    def _definition_menu(self, did, at):
        d = self.network.definitions[did]
        menu = QMenu(self)
        open_ = menu.addAction("Open (edit its contents)")
        response = menu.addAction("Analyse its response…")
        rename = menu.addAction("Rename…")
        duplicate = menu.addAction("Duplicate (an independent copy)")
        export = menu.addAction(f"Export as library block (*{BLOCK_SUFFIX})…")
        delete = menu.addAction("Delete from the library")
        chosen = menu.exec_(at)
        if chosen is response:
            block = new_block(SUBSYSTEM, name=d["name"])
            block.update(id="analysed", name=d["name"], definition=did)
            QTimer.singleShot(0, lambda: self.analyse_response(block, in_network=False))
        elif chosen is open_:
            instance = self._find_instance(did)
            if instance is None:
                QMessageBox.information(self, TITLE, "Place the subsystem on the canvas first, then double-click it.")
            else:
                self.path = instance
                self._after_move()
        elif chosen is rename:
            name, ok = QInputDialog.getText(self, TITLE, "Subsystem name:", text=d["name"])
            if ok and name.strip():
                d["name"] = name.strip()
                self._set_dirty()
                self._fill_library()
                self._show_location()
                self.scene.update()
        elif chosen is duplicate:
            self.network.duplicate_definition(did)
            self._set_dirty()
            self._fill_library()
        elif chosen is export:
            self.export_block(did)
        elif chosen is delete:
            try:
                self.network.remove_definition(did)
            except ValueError as exc:
                QMessageBox.information(self, TITLE, str(exc))
                return
            self._set_dirty()
            self._fill_library()

    def _find_instance(self, did):
        """The path (instance ids) to the first instance of a definition, searching from the top level."""
        def search(container, path, depth):
            for b in container["blocks"]:
                if b["type"] == SUBSYSTEM:
                    if b.get("definition") == did:
                        return path + [b["id"]]
                    inner = self.network.definition_of(b)
                    if inner is not None and depth < 32:
                        found = search(inner, path + [b["id"]], depth + 1)
                        if found:
                            return found
            return None
        return search(self.network.root, [], 0)

    def add_component_dialog(self, kind):
        pattern = "Characterised full neuron (*.mfn)" if kind == NEURON else "Valve design (*.mad)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Add " + ("neurons" if kind == NEURON else "valves"),
                                                self._dialog_dir(), pattern)
        for path in paths:
            self.add_component(path)

    def add_component(self, path):
        self._remember_dir(path)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            component = component_from_file(path)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, TITLE, f"Could not add {Path(path).name}:\n{exc}")
            return None
        QApplication.restoreOverrideCursor()
        cid = self.network.add_component(component)
        c = self.network.components[cid]
        n = sum(1 for x in c["converged"] if x is not None)
        self.log_message(f"Added {c['name']} ({n} point(s); " +
                         ", ".join(f"{a['name']} {min(a['values']):.4g}…{max(a['values']):.4g} {a['unit']}"
                                   for a in c["axes"]) + ")" +
                         ("" if c.get("complete", True) else " - the characterisation is incomplete"))
        self.statuses[cid] = CURRENT
        self._set_dirty()
        self._fill_library()
        return cid

    def update_component(self, cid):
        try:
            self.network.update_component(cid)
        except Exception as exc:
            QMessageBox.warning(self, TITLE, f"Could not update:\n{exc}")
            return
        self.statuses[cid] = CURRENT
        self.log_message(f"Updated {self.network.components[cid]['name']} from its source.")
        self._set_dirty()
        self._fill_library()
        self._rebuild_scene()

    def import_block_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import library block", self._dialog_dir(), BLOCK_FILTER)
        if path:
            self.import_block(path)

    def import_block(self, path):
        self._remember_dir(path)
        try:
            did = self.network.import_definition(path)
        except Exception as exc:
            QMessageBox.warning(self, TITLE, f"Could not import {Path(path).name}:\n{exc}")
            return None
        self.statuses = {cid: source_status(c) for cid, c in self.network.components.items()}
        self.log_message(f"Imported the library block {self.network.definitions[did]['name']} from {Path(path).name}.")
        self._set_dirty()
        self._fill_library()
        return did

    def export_block(self, did):
        d = self.network.definitions[did]
        path, _ = QFileDialog.getSaveFileName(self, "Export library block",
                                              str(Path(self._dialog_dir()) / (d["name"] + BLOCK_SUFFIX)), BLOCK_FILTER)
        if not path:
            return
        self.network.export_definition(did, path)
        self._remember_dir(path)
        self.log_message(f"Exported {d['name']} to {path} (with the components and subsystems it uses).")

    def check_sources(self):
        if not self.isVisible() or not self.network.components:
            return
        statuses = {cid: source_status(c) for cid, c in self.network.components.items()}
        if statuses != self.statuses:
            for cid, s in statuses.items():
                if s != self.statuses.get(cid) and s != CURRENT:
                    self.log_message(f"{self.network.components[cid]['name']}: source {s} "
                                     "(right-click it in the library to update)")
            self.statuses = statuses
            self._fill_library()

    # -----------------------------
    # Scene
    # -----------------------------

    def _rebuild_scene(self):
        self._building = True
        self.scene.clear()
        self.items, self.wire_items = {}, []
        container = self.container()
        for block in container["blocks"]:
            item = BlockItem(self, block)
            self.scene.addItem(item)
            self.items[block["id"]] = item
        for wire in container["wires"]:
            self._add_wire_item(wire)
        self._building = False
        self._fill_run()
        self._changed()
        self._show_block()

    def _add_wire_item(self, wire):
        src_item, dst_item = self.items.get(wire["from"][0]), self.items.get(wire["to"][0])
        if src_item is None or dst_item is None:
            return
        src, dst = src_item.port(wire["from"][1], True), dst_item.port(wire["to"][1], False)
        if src is None or dst is None:
            return
        ideal = not (src_item.block["type"] == NEURON and not wire["from"][1].startswith("out:"))
        item = WireItem(wire, src, dst, ideal)
        self.scene.addItem(item)
        self.wire_items.append(item)

    def _refresh_block(self, block):
        """Rebuild one block after its ports changed (wires on removed ports are dropped)."""
        removed = self.network.prune_wires()
        if removed:
            self.log_message(f"{removed} wire(s) removed with their ports.")
        for item in self.wire_items:
            item.detach()
            self.scene.removeItem(item)
        self.wire_items = []
        self.items[block["id"]].rebuild()
        for wire in self.container()["wires"]:
            self._add_wire_item(wire)

    def add_block(self, kind, ref=None, pos=QPointF(0, 0)):
        """A block from the library: ref = a component id (neuron, valve) or a subsystem definition id."""
        container = self.container()
        try:
            if kind == SUBSYSTEM:
                block = self.network.add_block(SUBSYSTEM, pos=(snap(pos.x()), snap(pos.y())), container=container,
                                               definition=ref)
            else:
                block = self.network.add_block(kind, ref, (snap(pos.x()), snap(pos.y())), container=container)
        except ValueError as exc:
            QMessageBox.information(self, TITLE, str(exc))
            return None
        item = BlockItem(self, block)
        self.scene.addItem(item)
        self.items[block["id"]] = item
        self.scene.clearSelection()
        item.setSelected(True)
        self._set_dirty()
        if kind in (INPUT, SUBSYSTEM):
            self._fill_run()
        if kind in (SUBSYSTEM, INPORT, OUTPORT):
            self._fill_library()
        self._changed()
        return block

    def connect_ports(self, src, dst):
        container = self.container()
        try:
            replaced = [w for w in container["wires"] if w["to"] == list(dst)]
            wire = self.network.connect(src, dst, container)
        except ValueError as exc:
            self.log_message(str(exc))
            return
        for item in [w for w in self.wire_items if w.wire in replaced]:
            item.detach()
            self.scene.removeItem(item)
            self.wire_items.remove(item)
        self._add_wire_item(wire)
        for text in self.network.unit_warnings(container):
            self.log_message("Warning: " + text)
        self._set_dirty()
        self._changed()

    def delete_selected(self):
        selected = self.scene.selectedItems()
        if not selected:
            return
        container = self.container()
        for item in selected:
            if isinstance(item, WireItem) and item in self.wire_items:
                self.network.disconnect(item.wire, container)
        for item in selected:
            if isinstance(item, BlockItem):
                self.network.remove_block(item.block["id"], container)
        self._set_dirty()
        self._fill_library()
        self._rebuild_scene()

    def group_selected(self, name=None):
        ids = [i.block["id"] for i in self.scene.selectedItems() if isinstance(i, BlockItem)]
        if not ids:
            QMessageBox.information(self, TITLE, "Select the blocks to put into a subsystem (drag a box around them).")
            return
        if name is None:
            name, ok = QInputDialog.getText(self, TITLE, "Name of the new subsystem:", text="Subsystem")
            if not ok:
                return
        try:
            sub = self.network.group(ids, self.container(), name.strip() or "Subsystem")
        except ValueError as exc:
            QMessageBox.information(self, TITLE, str(exc))
            return
        sub["pos"] = [snap(sub["pos"][0]), snap(sub["pos"][1])]
        d = self.network.definition_of(sub)
        self.log_message(f"Created the subsystem {d['name']} from {len(ids)} block(s): "
                         f"{sum(b['type'] == INPORT for b in d['blocks'])} in port(s), "
                         f"{sum(b['type'] == OUTPORT for b in d['blocks'])} out port(s). Double-click it to open it.")
        self._set_dirty()
        self._fill_library()
        self._rebuild_scene()
        if sub["id"] in self.items:
            self.items[sub["id"]].setSelected(True)

    def block_menu(self, block, at):
        menu = QMenu(self)
        c = self.network.component_of(block)
        exists = bool(c and c.get("source")) and Path(c["source"]).exists()
        analyse = open_design = open_sub = response = None
        if block["type"] not in (INPUT, CONSTANT, OUTPUT, INPORT, OUTPORT):
            response = menu.addAction("Analyse the block's response…")
            response.setToolTip("Evaluate this block on its own over its inputs (and parameters) and plot it in the "
                                "Analysis tab")
        if block["type"] == NEURON:
            analyse = menu.addAction("Open its characterisation in Analysis…")
            analyse.setEnabled(exists)
            if not exists:
                analyse.setToolTip("The neuron's .mfn was not found")
        elif block["type"] == VALVE:
            open_design = menu.addAction("Open the design…")
            open_design.setEnabled(exists)
        elif block["type"] == SUBSYSTEM:
            open_sub = menu.addAction("Open subsystem")
        menu.addSeparator()
        group = menu.addAction("Create subsystem from selection (Ctrl+G)")
        delete = menu.addAction("Delete")
        chosen = menu.exec_(at)
        if chosen is None:
            return
        if chosen is response:
            QTimer.singleShot(0, lambda: self.analyse_response(block))
        elif chosen is analyse:
            self.analyse_block(block)
        elif chosen is open_design:
            self.open_source.emit(c["source"])
        elif chosen is open_sub:
            QTimer.singleShot(0, lambda: self.enter(block))
        elif chosen is group:
            QTimer.singleShot(0, lambda: self.group_selected())
        elif chosen is delete:
            QTimer.singleShot(0, self.delete_selected)

    def _held_inputs(self, block):
        """The present value of each of a block's input ports in the evaluated network (where it is wired)."""
        values = (self.result or {}).get("values", {})
        held, prefix = {}, self.prefix()
        container = self.container()
        for w in container["wires"]:
            if w["to"][0] == block["id"]:
                v = values.get((prefix + w["from"][0], w["from"][1]))
                if v is not None and math.isfinite(v):
                    held[w["to"][1]] = v
        if block["type"] in (NEURON, VALVE):
            for port in self.network.ports(block)[0]:
                held.setdefault(port, float(block["axes"].get(port, {}).get("value", 0.0)))
        return held

    def analyse_response(self, block, in_network=True):
        """Choose what to sweep, evaluate the block on its own and show it in the Analysis tab."""
        held = self._held_inputs(block) if in_network else {}
        dialog = AnalyseDialog(self, self.network, block, held)
        if not dialog.exec_():
            return
        sweeps, held, params = dialog.sweeps()
        if not sweeps:
            QMessageBox.information(self, TITLE, "Tick at least one variable to sweep.")
            return
        n = int(np.prod([len(s["values"]) for s in sweeps]))
        progress = QProgressDialog(f"Evaluating {block['name']}…", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(400)

        def step(done, total):
            if done % 50 == 0 or done == total:
                progress.setValue(done)
                QApplication.processEvents()
            return not progress.wasCanceled()

        saved = [(prm, prm["value"]) for prm, _ in params]
        try:
            for prm, v in params:          # held parameters at the dialog's values (restored below)
                self.network.set_parameter(prm, v)
            data = self.network.analyse_block(block, sweeps, held, step)
        except Exception as exc:
            progress.close()
            QMessageBox.warning(self, TITLE, f"Could not analyse {block['name']}:\n{exc}")
            return
        finally:
            for prm, v in saved:
                self.network.set_parameter(prm, v)
        progress.close()
        if not data["complete"]:
            self.log_message(f"Analysis of {block['name']} stopped early: the rest is left empty.")
        names = [f"{s['label']}.{'input' if s['kind'] == 'input' else 'parameter'}" for s in sweeps]
        inputs = [nm for nm, s in zip(names, sweeps) if s["kind"] == "input"]
        axes = {nm: {"role": "input"} for nm in inputs[:2]}
        for nm, s in zip(names, sweeps):
            if nm not in axes:          # the slider starts at its present value in the network
                value = s.get("present", s["values"][len(s["values"]) // 2])
                axes[nm] = {"role": "constant", "value": float(value)}
        if not inputs:
            for nm in names[:2]:
                axes[nm] = {"role": "input"}
        self.log_message(f"Analysed {block['name']}: {n} evaluation(s) of " + ", ".join(s["label"] for s in sweeps))
        self.analyse_data.emit(data, {"axes": axes, "key": data["keys"][0]["key"], "block": block["name"]})

    def analyse_block(self, block):
        """Show a neuron block's characterisation in the Analysis tab: its input ports as the plot axes, its held
        variables at the block's values, its first output as Z."""
        c = self.network.component_of(block)
        if not c or not c.get("source") or not Path(c["source"]).exists():
            QMessageBox.information(self, TITLE, "The neuron's characterised file (*.mfn) was not found.")
            return
        if c.get("sha1") and source_status(c) == OUTDATED:
            self.log_message(f"Note: {Path(c['source']).name} was characterised again since it was added here; the "
                             "Analysis tab shows the file as it is now.")
        self.analyse.emit(c["source"], {"axes": {n: dict(s) for n, s in block["axes"].items()},
                                        "key": (block.get("outputs") or [None])[0], "block": block["name"]})

    def _open_selected(self):
        block = self.selected_block()
        if block is not None and block["type"] == SUBSYSTEM:
            self.enter(block)

    def _select_all(self):
        for item in self.items.values():
            item.setSelected(True)

    def fit_view(self):
        rect = self.scene.itemsBoundingRect()
        if rect.isNull():
            return
        self.view.fitInView(rect.adjusted(-60, -60, 60, 60), Qt.KeepAspectRatio)
        if self.view.transform().m11() > 1.5:
            self.view.resetTransform()
            self.view.centerOn(rect.center())

    def _selection_changed(self):
        if not self._building:
            QTimer.singleShot(0, self._show_block)

    def show_block_tab(self):
        self.tabs.setCurrentIndex(0)

    def selected_block(self):
        if self._building:
            return None
        try:
            blocks = [i.block for i in self.scene.selectedItems() if isinstance(i, BlockItem)]
        except RuntimeError:       # the scene is being cleared
            return None
        return blocks[0] if len(blocks) == 1 else None

    # -----------------------------
    # Block properties
    # -----------------------------

    def _show_block(self):
        block = self.selected_block()
        w = QWidget()
        layout = QVBoxLayout(w)
        container = self.container()
        if block is None:
            where = "this subsystem" if self.path else "the network"
            label = QLabel(f"{len(container['blocks'])} block(s) and {len(container['wires'])} wire(s) in {where}; "
                           f"{len(self.network.components)} component(s) and {len(self.network.definitions)} "
                           "subsystem(s) in the library.\n\nSelect one block to see its settings. Select several "
                           "and press Ctrl+G to make a subsystem of them.")
            label.setWordWrap(True)
            layout.addWidget(label)
            layout.addStretch(1)
            self.block_area.setWidget(w)
            return
        kind = block["type"]
        form = QFormLayout()
        name = QLineEdit(block["name"])
        name.editingFinished.connect(lambda: self._rename(block, name.text()))
        form.addRow(f"{TYPE_NAMES[kind]} name", name)
        if TYPES[kind].tip:
            tip = QLabel(TYPES[kind].tip)
            tip.setWordWrap(True)
            tip.setStyleSheet("color: #555;")
            form.addRow(tip)
        for p in TYPES[kind].params:
            form.addRow(p["label"], self._param_widget(block, p))
        if kind in (INPORT, OUTPORT):
            spin = QSpinBox()
            spin.setRange(1, 99)
            spin.setValue(int(block.get("index", 1)))
            spin.setToolTip("The port's position on the subsystem block (from the top)")
            spin.valueChanged.connect(lambda v: self._set(block, "index", int(v), ports=True))
            form.addRow("Port number", spin)
        layout.addLayout(form)
        if kind in (NEURON, VALVE):
            self._component_form(block, layout)
        elif kind == SUBSYSTEM:
            self._subsystem_form(block, layout)
        if kind not in (INPUT, CONSTANT, OUTPUT, INPORT, OUTPORT):
            b = QPushButton("Analyse the response…")
            b.setToolTip("Evaluate this block on its own over its inputs (and parameters) and plot it in the Analysis "
                         "tab")
            b.clicked.connect(lambda: QTimer.singleShot(0, lambda: self.analyse_response(block)))
            row = QHBoxLayout()
            row.addWidget(b)
            row.addStretch(1)
            layout.addLayout(row)
        layout.addStretch(1)
        self.block_area.setWidget(w)

    def _param_widget(self, block, p):
        key, kind = p["key"], p["kind"]
        if kind == "float":
            s = QDoubleSpinBox()
            s.setRange(-1e9, 1e9)
            s.setDecimals(4)
            s.setValue(float(block.get(key, p["default"])))
            s.setKeyboardTracking(False)
            s.valueChanged.connect(lambda v: self._set(block, key, float(v)))
            w = s
        elif kind == "int":
            s = QSpinBox()
            s.setRange(1, 24)
            s.setValue(int(block.get(key, p["default"])))
            s.valueChanged.connect(lambda v: self._set(block, key, int(v), ports=True))
            w = s
        elif kind == "choice":
            c = QComboBox()
            c.addItems(p["choices"])
            c.setCurrentText(str(block.get(key, p["default"])))
            c.currentTextChanged.connect(lambda t: self._set(block, key, t))
            w = c
        else:          # text, list
            e = QLineEdit(str(block.get(key, p["default"])))
            ports = key in ("signs", "ops")
            e.editingFinished.connect(lambda: self._set_text(block, p, e.text(), ports))
            w = e
        if p.get("tip"):
            w.setToolTip(p["tip"])
        if kind == "float" and block["type"] in FUNCTION_TYPES:     # a setting training may change, or not
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(w, 1)
            role = QComboBox()
            role.addItems(["Constant", "Parameter"])
            role.setCurrentIndex(1 if key in (block.get("trainable") or []) else 0)
            role.setToolTip("Parameter: training may change this value (◆ on the block). Constant: never trained.")
            role.currentIndexChanged.connect(lambda i: self._set_trainable(block, key, i == 1))
            h.addWidget(role)
            return holder
        return w

    def _set_trainable(self, block, key, on):
        keys = [k for k in (block.get("trainable") or []) if k != key] + ([key] if on else [])
        block["trainable"] = [q["key"] for q in TYPES[block["type"]].params if q["key"] in keys]
        self._refresh_block(block)
        self._fill_parameters()
        self._set_dirty()

    def _set_text(self, block, p, text, ports):
        text = text.strip()
        if p["key"] == "signs":
            text = "".join(ch for ch in text if ch in "+-") or "+"
        elif p["key"] == "ops":
            text = "".join(ch for ch in text if ch in "*/") or "*"
        elif p["kind"] == "list":
            try:
                [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
            except ValueError:
                QMessageBox.information(self, TITLE, f"{p['label']}: numbers separated by commas, e.g. 0, 10, 20.")
                return
        elif p["key"] == "expr":
            try:
                compile(text, "<function block>", "eval")
            except SyntaxError as exc:
                QMessageBox.information(self, TITLE, f"The expression is not valid:\n{exc}")
                return
        self._set(block, p["key"], text, ports=ports)

    def _subsystem_form(self, block, layout):
        did = block.get("definition")
        d = self.network.definition_of(block)
        if d is None:
            layout.addWidget(QLabel("The subsystem's definition is missing."))
            return
        n = len(self.network.instances(did))
        ins = [b["name"] for b in sorted((b for b in d["blocks"] if b["type"] == INPORT),
                                         key=lambda b: b.get("index", 0))]
        outs = [b["name"] for b in sorted((b for b in d["blocks"] if b["type"] == OUTPORT),
                                          key=lambda b: b.get("index", 0))]
        info = QLabel(f"Subsystem <b>{d['name']}</b>: {len(d['blocks'])} block(s), {len(d['wires'])} wire(s)."
                      f"<br>In ports: {', '.join(ins) or '-'}<br>Out ports: {', '.join(outs) or '-'}"
                      + (f"<br>{n} blocks use these contents: editing it changes all of them." if n > 1 else ""))
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QFormLayout()
        dname = QLineEdit(d["name"])
        dname.editingFinished.connect(lambda: self._rename_definition(d, dname.text()))
        form.addRow("Subsystem name", dname)
        layout.addLayout(form)
        row = QHBoxLayout()
        b = QPushButton("Open")
        b.clicked.connect(lambda: QTimer.singleShot(0, lambda: self.enter(block)))
        row.addWidget(b)
        b = QPushButton("Make independent")
        b.setToolTip("Give this block its own copy of the contents (other blocks keep the shared one)")
        b.setEnabled(n > 1)
        b.clicked.connect(lambda: self._make_independent(block))
        row.addWidget(b)
        b = QPushButton("Export…")
        b.setToolTip(f"Save as a library block (*{BLOCK_SUFFIX}) for other networks")
        b.clicked.connect(lambda: self.export_block(did))
        row.addWidget(b)
        layout.addLayout(row)

    def _rename_definition(self, d, text):
        text = text.strip()
        if text and text != d["name"]:
            d["name"] = text
            self._set_dirty()
            self._fill_library()
            self.scene.update()

    def _make_independent(self, block):
        new = self.network.duplicate_definition(block["definition"])
        block["definition"] = new
        self.log_message(f"{block['name']} now has its own copy: {self.network.definitions[new]['name']}.")
        self._set_dirty()
        self._fill_library()
        self._refresh_block(block)
        self.scene.update()
        QTimer.singleShot(0, self._show_block)

    def _rename(self, block, text):
        text = text.strip()
        if not text or text == block["name"]:
            return
        if any(b["name"] == text for b in self.container()["blocks"] if b is not block):
            QMessageBox.information(self, TITLE, f"There is already a block called {text} here.")
            QTimer.singleShot(0, self._show_block)
            return
        block["name"] = text
        self._refresh_block(block)
        self._set_dirty()
        if block["type"] in (INPUT, OUTPUT, SUBSYSTEM):
            self._fill_run()
        if block["type"] in (INPORT, OUTPORT):
            self._fill_library()
        self._changed()

    def _set(self, block, key, value, ports=False):
        if block.get(key) == value:
            return
        block[key] = value
        self._refresh_block(block)
        if key in (block.get("trainable") or []):
            QTimer.singleShot(0, self._fill_parameters)
        if block["type"] == INPUT and key == "value":
            self._fill_run()
        self._set_dirty()
        self._changed()

    def _component_form(self, block, layout):
        c = self.network.component_of(block)
        if c is None:
            layout.addWidget(QLabel("The block's component is missing from the library."))
            return
        status = self.statuses.get(block["component"], CURRENT)
        n = sum(1 for x in c["converged"] if x is not None)
        info = QLabel(f"<b>{c['name']}</b> ({'neuron' if c['kind'] == NEURON else 'valve'}), {n} point(s)"
                      + ("" if c.get("complete", True) else ", <b>incomplete</b>")
                      + (f"<br>Valve: {c['design']}" if c.get("design") else "")
                      + f"<br><span style='color:#555'>{c.get('source') or ''}</span>"
                      + (f"<br><span style='color:{STATUS_COLORS[status]}'>Source {status}</span>"
                         if status in STATUS_COLORS else ""))
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(info)

        box = QGroupBox("Variables (the characterisation's swept parameters)")
        v = QVBoxLayout(box)
        note = QLabel("<b>Input</b>: a port, wired to another block. <b>Parameter</b>: held at its value, which "
                      "training may change within the characterised range (◆ on the block). <b>Constant</b>: held "
                      "at its value, never trained.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        v.addWidget(note)
        table = QTableWidget(len(c["axes"]), 4)
        table.setHorizontalHeaderLabels(["Variable", "Characterised", "Use as", "Value"])
        table.verticalHeader().setVisible(False)
        for r, a in enumerate(c["axes"]):
            setting = block["axes"].setdefault(a["name"], {"role": ROLE_CONSTANT, "value": float(a["values"][0])})
            values = sorted(a["values"])
            for col, text in ((0, short_label(a["name"])),
                              (1, f"{values[0]:.4g} … {values[-1]:.4g} {a.get('unit', '')} ({len(values)})")):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemIsEnabled)
                cell.setToolTip(f"{a['name']}: {a.get('label', a['name'])} [{a.get('unit', '')}], simulated at "
                                + ", ".join(f"{x:.4g}" for x in values))
                table.setItem(r, col, cell)
            role = QComboBox()
            role.addItems(["Input", "Parameter", "Constant"])
            role.setCurrentIndex(ROLES.index(setting["role"]) if setting["role"] in ROLES else 2)
            role.currentIndexChanged.connect(lambda i, s=setting: self._set_role(block, s, ROLES[i]))
            table.setCellWidget(r, 2, role)
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(4)
            step = (values[-1] - values[0]) / max(len(values) - 1, 1)
            spin.setSingleStep(step if step > 0 else 0.1)
            spin.setValue(float(setting["value"]))
            spin.setKeyboardTracking(False)
            spin.setMaximumWidth(95)
            spin.setToolTip("Parameter or constant: the value it is held at. Input: the value used while the port "
                            "is not connected.")
            spin.valueChanged.connect(lambda val, s=setting: self._set_axis_value(block, s, val))
            table.setCellWidget(r, 3, spin)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.setTextElideMode(Qt.ElideMiddle)
        table.setWordWrap(False)
        table.setMinimumHeight(min(60 + 30 * len(c["axes"]), 330))
        v.addWidget(table)
        layout.addWidget(box)

        box = QGroupBox("Output ports (recorded quantities)")
        v = QVBoxLayout(box)
        outputs = QListWidget()
        for k in c["keys"]:
            item = QListWidgetItem(f"{k['key']}  [{k.get('unit', '')}]")
            item.setData(Qt.UserRole, k["key"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if k["key"] in block["outputs"] else Qt.Unchecked)
            outputs.addItem(item)
        outputs.setMaximumHeight(min(30 + 22 * len(c["keys"]), 220))
        outputs.itemChanged.connect(lambda _: self._set_outputs(block, outputs))
        v.addWidget(outputs)
        layout.addWidget(box)

        form = QFormLayout()
        mode = QComboBox()
        mode.addItems(["Extrapolate linearly", "Clamp to the characterised range"])
        mode.setCurrentIndex(0 if block.get("range", EXTRAPOLATE) == EXTRAPOLATE else 1)
        mode.currentIndexChanged.connect(lambda i: self._set(block, "range", EXTRAPOLATE if i == 0 else CLAMP))
        form.addRow("Outside the range", mode)
        layout.addLayout(form)
        row = QHBoxLayout()
        if c["kind"] == NEURON:
            b = QPushButton("Open its characterisation in Analysis…")
            b.setToolTip("Plot this neuron's stored characterisation in the Analysis tab (with the 3D view), at this "
                         "block's values")
            b.setEnabled(bool(c.get("source")) and Path(c["source"]).exists())
            b.clicked.connect(lambda: self.analyse_block(block))
            row.addWidget(b)
        else:
            b = QPushButton("Open the design…")
            b.setToolTip("Open the valve design in the Pre-activation → activation tab")
            b.setEnabled(bool(c.get("source")) and Path(c["source"]).exists())
            b.clicked.connect(lambda: self.open_source.emit(c["source"]))
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)
        flags = (self.result or {}).get("flags", {}).get(self.prefix() + block["id"])
        if flags:
            warn = QLabel("Now outside the characterised range: " + ", ".join(flags))
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #c00000;")
            layout.addWidget(warn)
        if c["kind"] == NEURON and any(k.startswith("P:") for k in block["outputs"]):
            note = QLabel("A chamber pressure wired into another block is an ideal signal: the receiving chamber "
                          "does not load this neuron (dashed wires).")
            note.setWordWrap(True)
            note.setStyleSheet("color: #555;")
            layout.addWidget(note)

    def _set_role(self, block, setting, role):
        if setting["role"] == role:
            return
        setting["role"] = role
        self._refresh_block(block)
        self._fill_parameters()
        self._set_dirty()
        self._changed()

    def _set_axis_value(self, block, setting, value):
        setting["value"] = float(value)
        self.items[block["id"]].update()
        self._set_dirty()
        self._changed()
        if setting.get("role") == ROLE_PARAMETER:
            QTimer.singleShot(0, self._fill_parameters)

    def _set_outputs(self, block, widget):
        block["outputs"] = [widget.item(i).data(Qt.UserRole) for i in range(widget.count())
                            if widget.item(i).checkState() == Qt.Checked]
        self._refresh_block(block)
        self._set_dirty()
        self._changed()

    # -----------------------------
    # Evaluation and sweeps
    # -----------------------------

    def _changed(self):
        if self.auto.isChecked():
            self.evaluate(quiet=True)

    def evaluate(self, quiet=False):
        if not self.network.blocks:
            for item in self.items.values():
                item.set_values({}, self.prefix(), None, {})
            return
        try:
            result = self.network.solve()
        except Exception as exc:
            self.log_message(f"Evaluation failed: {exc}")
            return
        self.result = result
        prefix = self.prefix()
        for item in self.items.values():
            units = {p: port.unit for p, port in item.outputs.items()}
            item.set_values(result["values"], prefix, result["flags"].get(prefix + item.block["id"]), units)
        probes = result["probes"]
        self.probe_label.setText("<br>".join(f"<b>{n}</b> = {_tidy(v):.6g}" if math.isfinite(v) else
                                             f"<b>{n}</b> = - (not connected)" for n, v in probes.items()) or
                                 "Add Output blocks to probe signals.")
        bits = []
        if result["loop"]:
            bits.append(f"algebraic loop ({len(result['loop'])} block(s)) "
                        + ("solved" if result["converged"] else f"NOT converged (residual {result['residual']:.3g})"))
        if result["flags"]:
            bits.append(f"{len(result['flags'])} block(s) outside their characterised range or failing")
        nan = [n for n, v in probes.items() if not math.isfinite(v)]
        self.status_label.setText("Evaluated" + (": " + "; ".join(bits) if bits else "") +
                                  (f". Not a number: {', '.join(nan)}" if nan else ""))
        if not quiet:
            self.log_message("Evaluated: " + ", ".join(f"{n} = {v:.6g}" for n, v in probes.items())
                             + ("; " + "; ".join(bits) if bits else ""))

    def _fill_run(self):
        while self.inputs_form.rowCount():
            self.inputs_form.removeRow(0)
        try:
            inputs = self.network.inputs()
        except ValueError:
            inputs = []
        for name, block in inputs:
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(4)
            spin.setValue(float(block.get("value", 0.0)))
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda v, b=block: self._set_input(b, v))
            self.inputs_form.addRow(f"{name} [{block.get('unit', '')}]", spin)
        if not inputs:
            self.inputs_form.addRow(QLabel("Add Input blocks (library → Sources and probes)."))
        names = [name for name, _ in inputs]
        sweep = self.network.sweep
        for combo, current, none in ((self.x_combo, sweep.get("x"), False), (self.y_combo, sweep.get("y"), True)):
            combo.blockSignals(True)
            combo.clear()
            if none:
                combo.addItem("(none)")
            combo.addItems(names)
            index = combo.findText(current) if current else -1
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)
        self._sweep_x_changed()
        self._sweep_y_changed()
        self._fill_parameters()

    def _fill_parameters(self):
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        params = self.network.parameters()
        for prm in params:
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            if prm["min"] is not None:
                spin.setRange(prm["min"], prm["max"])
                spin.setSingleStep((prm["max"] - prm["min"]) / 20 or 0.1)
            else:
                spin.setRange(-1e9, 1e9)
            spin.setValue(prm["value"])
            spin.setKeyboardTracking(False)
            spin.setToolTip(f"In {prm['where']}" + (f"; range {prm['min']:.4g} … {prm['max']:.4g} {prm['unit']}"
                                                     if prm["min"] is not None else "; unbounded"))
            spin.valueChanged.connect(lambda v, q=prm: self._set_parameter(q, v))
            where = "" if prm["where"] == "top level" else f" ({prm['where']})"
            unit = f" [{prm['unit']}]" if prm["unit"] else ""
            self.params_form.addRow(f"◆ {prm['name']}{where}{unit}", spin)
        if not params:
            note = QLabel("None yet: set a neuron's variable, or a setting of a math or activation block, to "
                          "Parameter.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #555;")
            self.params_form.addRow(note)

    def _set_parameter(self, prm, value):
        self.network.set_parameter(prm, value)
        item = self.items.get(prm["block"]["id"])
        if item is not None and item.block is prm["block"]:
            item.rebuild() if prm["kind"] == "setting" else item.update()
            for wire in self.wire_items:
                wire.update_path()
        if self.selected_block() is prm["block"]:
            QTimer.singleShot(0, self._show_block)
        self._set_dirty()
        self._changed()

    def _set_input(self, block, value):
        block["value"] = float(value)
        self._set_dirty()
        if self.selected_block() is block:
            QTimer.singleShot(0, self._show_block)
        self._changed()

    def _input_block(self, name):
        try:
            return next((b for n, b in self.network.inputs() if n == name), None)
        except ValueError:
            return None

    def _sweep_x_changed(self):
        block = self._input_block(self.x_combo.currentText())
        self.network.sweep["x"] = self.x_combo.currentText() if block else None
        for s in (self.x_from, self.x_to, self.x_points):
            s.blockSignals(True)
        if block:
            sw = block.setdefault("sweep", {"from": -20.0, "to": 20.0, "points": 41})
            self.x_from.setValue(sw["from"])
            self.x_to.setValue(sw["to"])
            self.x_points.setValue(int(sw["points"]))
        for s in (self.x_from, self.x_to, self.x_points):
            s.blockSignals(False)
            s.setEnabled(block is not None)

    def _sweep_x_edited(self):
        block = self._input_block(self.x_combo.currentText())
        if block:
            block["sweep"] = {"from": self.x_from.value(), "to": self.x_to.value(), "points": self.x_points.value()}
            self._set_dirty()

    def _sweep_y_changed(self):
        block = self._input_block(self.y_combo.currentText())
        self.network.sweep["y"] = self.y_combo.currentText() if block else None
        sweep = self.network.sweep
        for s in (self.y_from, self.y_to, self.y_points):
            s.blockSignals(True)
        self.y_from.setValue(sweep.get("y_from", -20.0))
        self.y_to.setValue(sweep.get("y_to", 20.0))
        self.y_points.setValue(int(sweep.get("y_points", 5)))
        for s in (self.y_from, self.y_to, self.y_points):
            s.blockSignals(False)
            s.setEnabled(block is not None)

    def _sweep_y_edited(self):
        self.network.sweep.update(y_from=self.y_from.value(), y_to=self.y_to.value(), y_points=self.y_points.value())
        self._set_dirty()

    def run_sweep(self):
        x = self.x_combo.currentText() if self._input_block(self.x_combo.currentText()) else None
        y = self.y_combo.currentText() if self._input_block(self.y_combo.currentText()) else None
        if not x:
            QMessageBox.information(self, TITLE, "Add an Input block and choose it as the sweep's x.")
            return
        if y == x:
            QMessageBox.information(self, TITLE, "Choose two different inputs for x and y.")
            return
        if not self.network.outputs():
            QMessageBox.information(self, TITLE, "Add Output blocks: they are what a sweep records and plots.")
            return
        xs = np.linspace(self.x_from.value(), self.x_to.value(), self.x_points.value())
        ys = np.linspace(self.y_from.value(), self.y_to.value(), self.y_points.value()) if y else None
        total = len(xs) * (len(ys) if y else 1)
        dialog = QProgressDialog("Sweeping the network…", "Cancel", 0, total, self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(400)

        def progress(done, n):
            if done % 20 == 0 or done == n:
                dialog.setValue(done)
                QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            sweep = self.network.run_sweep(x, xs, y, ys, progress)
        except Exception as exc:
            dialog.close()
            QMessageBox.warning(self, TITLE, f"The sweep failed:\n{exc}")
            return
        dialog.close()
        self.sweep_result = sweep
        bad, flagged = int((~sweep["converged"]).sum()), int(sweep["flagged"].sum())
        self.log_message(f"Sweep of {x}" + (f" × {y}" if y else "") + f": {total} point(s)"
                         + (" (cancelled)" if sweep["cancelled"] else "")
                         + (f", {flagged} outside a block's characterised range" if flagged else "")
                         + (f", {bad} loop(s) not converged" if bad else ""))
        self.plot_sweep()
        self.bottom.setCurrentIndex(0)
        self.evaluate(quiet=True)

    def plot_sweep(self):
        sweep = self.sweep_result
        self.figure.clear()
        if sweep is None:
            self.canvas.draw_idle()
            return
        names = list(sweep["probes"])
        xs = sweep["x_values"]
        unit = lambda name: (self._input_block(name) or {}).get("unit", "")  # noqa: E731
        xlabel = f"{sweep['x']} [{unit(sweep['x'])}]"
        if sweep["y"] is None or len(sweep["y_values"]) <= MAX_LINES:
            ys = sweep["y_values"] if sweep["y"] else [None]
            # without y every Output is a curve in one panel; with y each Output gets a panel of curves (one per y)
            groups = [names] if sweep["y"] is None else [[n] for n in names]
            for k, group in enumerate(groups or [[]]):
                ax = self.figure.add_subplot(1, len(groups or [[]]), k + 1)
                for n in group:
                    for j, yv in enumerate(ys):
                        z = sweep["probes"][n][j]
                        label = f"{sweep['y']} = {yv:.4g}" if yv is not None else n
                        ax.plot(xs, z, label=label)
                        out = sweep["flagged"][j]
                        if out.any():
                            ax.plot(xs[out], z[out], "o", ms=3, color="#d32f2f")
                        bad = ~sweep["converged"][j]
                        if bad.any():
                            ax.plot(xs[bad], z[bad], "x", color="black")
                if sweep["y"] is not None and group:
                    ax.set_title(group[0], fontsize=9)
                ax.set_xlabel(xlabel)
                if k == 0:
                    ax.set_ylabel("output")
                ax.grid(True, alpha=0.3)
                if group and (k == 0 or sweep["y"] is None):
                    ax.legend(fontsize=7)
        else:
            n = max(len(names), 1)
            for k, name in enumerate(names):
                ax = self.figure.add_subplot(1, n, k + 1)
                mesh = ax.pcolormesh(xs, sweep["y_values"], sweep["probes"][name], shading="auto", cmap="viridis")
                self.figure.colorbar(mesh, ax=ax)
                ax.set_title(name, fontsize=9)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(f"{sweep['y']} [{unit(sweep['y'])}]")
                X, Y = np.meshgrid(xs, sweep["y_values"])
                if sweep["flagged"].any():
                    ax.plot(X[sweep["flagged"]], Y[sweep["flagged"]], ".", ms=2, color="#d32f2f")
        self.canvas.draw_idle()

    def export_csv(self):
        if self.sweep_result is None:
            QMessageBox.information(self, TITLE, "Run a sweep first.")
            return
        stem = Path(self.network.path).stem if self.network.path else "network"
        path, _ = QFileDialog.getSaveFileName(self, "Export sweep", str(Path(self._dialog_dir()) / f"{stem}_sweep.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        Path(path).write_text(sweep_csv(self.sweep_result), encoding="utf-8")
        self._remember_dir(path)
        self.log_message(f"Exported {path}")

    # -----------------------------
    # Files
    # -----------------------------

    def _dialog_dir(self):
        return QSettings("MembraneNeuronSimulator", "App").value("last_dir", str(Path.cwd()))

    def _remember_dir(self, path):
        QSettings("MembraneNeuronSimulator", "App").setValue("last_dir", str(Path(path).parent))

    def _confirm_discard(self):
        if not self.dirty or not self.network.blocks:
            return True
        answer = QMessageBox.question(self, TITLE, "Discard the changes to the current network?",
                                      QMessageBox.Discard | QMessageBox.Cancel)
        return answer == QMessageBox.Discard

    def _set_network(self, network):
        self.network = network
        self.path = []
        self.result = self.sweep_result = None
        self.statuses = {cid: source_status(c) for cid, c in network.components.items()}
        self.dirty = False
        self._show_location()
        self._fill_library()
        self._rebuild_scene()
        self.plot_sweep()
        self._update_title()
        QTimer.singleShot(0, self.fit_view)

    def new_network(self):
        if self._confirm_discard():
            self._set_network(Network())
            self.log_message("New network.")

    def open_project(self, path=None):
        if not path:
            if not self._confirm_discard():
                return
            path, _ = QFileDialog.getOpenFileName(self, "Open network", self._dialog_dir(), FILTER)
            if not path:
                return
        self._remember_dir(path)
        try:
            network = Network.load(path)
        except Exception as exc:
            QMessageBox.critical(self, TITLE, f"Could not open the network:\n{exc}")
            return
        self._set_network(network)
        self.log_message(f"Opened {Path(path).name}: {len(network.blocks)} block(s), {len(network.wires)} wire(s), "
                         f"{len(network.definitions)} subsystem(s).")
        for cid, s in self.statuses.items():
            if s != CURRENT:
                self.log_message(f"{network.components[cid]['name']}: source {s} (the stored copy is used)")

    def save_project(self, save_as=False):
        path = self.network.path
        if save_as or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save network", str(Path(self._dialog_dir()) /
                                                                         ("network" + SUFFIX)), FILTER)
            if not path:
                return
        self.network.save(path)
        self._remember_dir(path)
        self.dirty = False
        self._update_title()
        self.log_message(f"Saved {path}")
