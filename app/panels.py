"""Dock panels: model tree, part properties, solver settings, results."""
from qtpy.QtCore import QLocale, QSettings, Qt, Signal
from qtpy.QtGui import QColor, QDoubleValidator, QIcon, QPixmap
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                            QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                            QListWidgetItem, QMenu, QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
                            QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .project import (AUTOMATIC, CHAMBER, DEFORMABLE, ROLE_COLORS, ROLE_FIELDS, ROLE_HELP, ROLES, SOLVER_FIELDS,
                      part_color)
from .viewport import RESULT_FIELDS


def color_icon(color, size=12):
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


def role_icon(role, size=12):
    return color_icon(ROLE_COLORS[role], size)


def fmt(value):
    return f"{value:.6g}"


# -----------------------------
# Model tree
# -----------------------------

class ModelTree(QTreeWidget):
    selection_changed = Signal(list)
    visibility_changed = Signal(int, bool)
    role_requested = Signal(list, str)
    show_only_requested = Signal(list)
    show_all_requested = Signal()

    def __init__(self, parent=None, roles=ROLES):
        super().__init__(parent)
        self.roles = list(roles)
        self.setHeaderLabels(["Part", "Role"])
        self.setRootIsDecorated(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.itemSelectionChanged.connect(self._emit_selection)
        self.itemChanged.connect(self._item_changed)
        self.customContextMenuRequested.connect(self._context_menu)
        self._updating = False

    def populate(self, parts):
        self._updating = True
        self.clear()
        for i, part in enumerate(parts):
            item = QTreeWidgetItem([part.name, part.role])
            item.setData(0, Qt.UserRole, i)
            item.setIcon(1, color_icon(part_color(part)))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if part.visible else Qt.Unchecked)
            item.setToolTip(0, "Tick to show, untick to hide")
            self.addTopLevelItem(item)
        self._updating = False

    def set_connections(self, connections):
        """Extra items under a 'Flow connections' group: [(label, detail, colour)]; selecting one emits the
        negative index -1 - k."""
        self._updating = True
        for i in reversed(range(self.topLevelItemCount())):
            if self.topLevelItem(i).data(0, Qt.UserRole) is None:
                self.takeTopLevelItem(i)
        if connections:
            group = QTreeWidgetItem(["Flow connections", ""])
            group.setFlags(Qt.ItemIsEnabled)
            group.setData(0, Qt.UserRole, None)
            font = group.font(0)
            font.setBold(True)
            group.setFont(0, font)
            for k, (label, detail, color) in enumerate(connections):
                item = QTreeWidgetItem([label, detail])
                item.setData(0, Qt.UserRole, -1 - k)
                item.setIcon(1, color_icon(color))
                item.setToolTip(0, "Flow connection: click to set its type and resistance (Flow tab)")
                group.addChild(item)
            self.addTopLevelItem(group)
            group.setExpanded(True)
        self._updating = False

    def refresh(self, parts):
        self._updating = True
        for i in range(min(self.topLevelItemCount(), len(parts))):
            item, part = self.topLevelItem(i), parts[i]
            item.setText(1, part.role)
            item.setIcon(1, color_icon(part_color(part)))
            item.setCheckState(0, Qt.Checked if part.visible else Qt.Unchecked)
        self._updating = False

    def select(self, indices):
        self._updating = True
        for i in range(self.topLevelItemCount()):
            top = self.topLevelItem(i)
            top.setSelected(top.data(0, Qt.UserRole) in indices)
            for c in range(top.childCount()):
                top.child(c).setSelected(top.child(c).data(0, Qt.UserRole) in indices)
        parts = [i for i in indices if i >= 0]
        if parts and self.topLevelItem(min(parts)) is not None:
            self.scrollToItem(self.topLevelItem(min(parts)))
        self._updating = False

    def selected_indices(self):
        return sorted(item.data(0, Qt.UserRole) for item in self.selectedItems()
                      if item.data(0, Qt.UserRole) is not None)

    def _emit_selection(self):
        if not self._updating:
            self.selection_changed.emit(self.selected_indices())

    def _item_changed(self, item, column):
        if not self._updating and column == 0 and (item.data(0, Qt.UserRole) or 0) >= 0:
            self.visibility_changed.emit(item.data(0, Qt.UserRole), item.checkState(0) == Qt.Checked)

    def _context_menu(self, pos):
        indices = [i for i in self.selected_indices() if i >= 0]
        if not indices:
            return
        menu = QMenu(self)
        assign = menu.addMenu("Assign role")
        for role in self.roles:
            action = assign.addAction(role_icon(role), role)
            action.triggered.connect(lambda _=False, r=role: self.role_requested.emit(indices, r))
        menu.addSeparator()
        menu.addAction("Show only these").triggered.connect(lambda: self.show_only_requested.emit(indices))
        menu.addAction("Show all").triggered.connect(self.show_all_requested.emit)
        menu.exec(self.viewport().mapToGlobal(pos))


# -----------------------------
# Generic property form
# -----------------------------

SEGMENT_TAG = "   (segment "


class OutputsEditor(QWidget):
    """Named outputs of a fluid: [{"name", "segment"}]. Choosing a segment (or an output in the list) asks for it
    to be highlighted; names are edited in place (double-click)."""
    changed = Signal(list)
    highlight = Signal(object)  # segment (1..N) or None

    def __init__(self, outputs, segments, parent=None):
        super().__init__(parent)
        self.outputs = [dict(o) for o in (outputs or ())]
        self.segments = max(int(segments or 1), 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Segment"))
        self.segment = QSpinBox()
        self.segment.setRange(1, self.segments)
        self.segment.setSuffix(f" of {self.segments}")
        self.segment.setToolTip("1 = the upstream end of the tube. The chosen segment is highlighted in the 3D view.")
        self.segment.valueChanged.connect(lambda k: self.highlight.emit(k))
        row.addWidget(self.segment)
        self.name = QLineEdit()
        self.name.setPlaceholderText("output name")
        self.name.returnPressed.connect(self._add)
        row.addWidget(self.name, 1)
        add = QPushButton("Add output")
        add.clicked.connect(self._add)
        row.addWidget(add)
        layout.addLayout(row)
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.list.setToolTip("Double-click a name to rename it.")
        self.list.currentRowChanged.connect(self._selected)
        self.list.itemChanged.connect(self._renamed)
        layout.addWidget(self.list)
        remove = QPushButton("Remove output")
        remove.clicked.connect(self._remove)
        layout.addWidget(remove, 0, Qt.AlignLeft)
        self._fill()

    def _fill(self):
        self.list.blockSignals(True)
        self.list.clear()
        for _ in self.outputs:
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._labels()

    def _labels(self):
        self.list.blockSignals(True)
        for k, o in enumerate(self.outputs):
            item = self.list.item(k)
            item.setText(f"{o['name']}{SEGMENT_TAG}{o['segment']})")
            item.setToolTip(f"{o['name']}: gas pressure of segment {o['segment']} of {self.segments}")
        self.list.blockSignals(False)

    def _add(self):
        k = self.segment.value()
        name = self.name.text().strip() or f"segment {k}"
        taken = {o["name"] for o in self.outputs}
        base, n = name, 2
        while name in taken:
            name, n = f"{base} ({n})", n + 1
        self.outputs.append({"name": name, "segment": k})
        self.name.clear()
        self._fill()
        self.list.setCurrentRow(len(self.outputs) - 1)
        self.changed.emit([dict(o) for o in self.outputs])

    def _remove(self):
        k = self.list.currentRow()
        if 0 <= k < len(self.outputs):
            del self.outputs[k]
            self._fill()
            self.changed.emit([dict(o) for o in self.outputs])
            self.highlight.emit(None)

    def _selected(self, k):
        if 0 <= k < len(self.outputs):
            self.segment.blockSignals(True)
            self.segment.setValue(min(self.outputs[k]["segment"], self.segments))
            self.segment.blockSignals(False)
            self.highlight.emit(self.outputs[k]["segment"])

    def _renamed(self, item):
        k = self.list.row(item)
        name = item.text().split(SEGMENT_TAG)[0].strip()
        if not (0 <= k < len(self.outputs)) or not name or name == self.outputs[k]["name"]:
            self._labels()
            return
        if name in {o["name"] for i, o in enumerate(self.outputs) if i != k}:
            self._labels()  # names must be unique
            return
        self.outputs[k]["name"] = name
        self._labels()
        self.changed.emit([dict(o) for o in self.outputs])


class FieldForm(QWidget):
    """Form generated from project.Field descriptors, editing a dict (or several dicts at once)."""
    edited = Signal(str, object)  # key, value
    highlight = Signal(object)    # a segment to highlight (outputs editor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_ = QFormLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    def build(self, fields, values: dict, context: dict = None):
        # Detach and delete the old rows *later*: build() can be triggered from one of these
        # widgets' own signals, and deleting a widget inside its signal crashes Qt.
        while self.layout_.rowCount():
            row = self.layout_.takeRow(0)
            for item in (row.labelItem, row.fieldItem):
                if item is not None and item.widget() is not None:
                    item.widget().blockSignals(True)
                    item.widget().hide()
                    item.widget().deleteLater()
        context = dict(context or {}, **values)
        for f in fields:
            if f.visible_if and context.get(f.visible_if[0]) not in f.visible_if[1]:
                continue
            value = values.get(f.key, f.default)
            if f.kind == "choice":
                widget = QComboBox()
                # choices named by a context key are filled at build time (e.g. the project's chambers)
                choices = [AUTOMATIC] + list(context.get(f.choices, ())) if isinstance(f.choices, str) else f.choices
                widget.addItems(list(choices))
                if widget.findText(str(value)) < 0:  # e.g. a chamber that was renamed or reassigned
                    widget.addItem(str(value))
                widget.setCurrentText(str(value))
                widget.currentTextChanged.connect(lambda v, key=f.key: self.edited.emit(key, v))
            elif f.kind == "outputs":
                widget = OutputsEditor(value, values.get("segments", 1))
                widget.changed.connect(lambda v, key=f.key: self.edited.emit(key, v))
                widget.highlight.connect(self.highlight.emit)
            elif f.kind == "file":
                widget = QWidget()
                h = QHBoxLayout(widget)
                h.setContentsMargins(0, 0, 0, 0)
                line = QLineEdit(str(value))
                line.setToolTip(str(value))
                line.editingFinished.connect(lambda w=line, key=f.key: self.edited.emit(key, w.text().strip()))
                browse = QPushButton("…")
                browse.setFixedWidth(28)
                browse.clicked.connect(lambda _, w=line, key=f.key, flt=";;".join(f.choices):
                                       self._browse(key, w, flt))
                h.addWidget(line, 1)
                h.addWidget(browse)
            elif f.kind == "text":
                widget = QLineEdit(str(value))
                widget.editingFinished.connect(lambda w=widget, key=f.key: self.edited.emit(key, w.text()))
            elif f.kind == "int":
                widget = QSpinBox()
                widget.setRange(int(f.minimum), int(f.maximum))
                widget.setValue(int(value))
                widget.editingFinished.connect(lambda w=widget, key=f.key: self.edited.emit(key, w.value()))
            else:
                widget = QLineEdit(fmt(float(value)) if value is not None else "")
                validator = QDoubleValidator(f.minimum, f.maximum, 12)
                validator.setNotation(QDoubleValidator.ScientificNotation)
                validator.setLocale(QLocale.c())
                widget.setValidator(validator)
                widget.editingFinished.connect(lambda w=widget, key=f.key: self._float_edited(key, w))
            if f.tooltip:
                widget.setToolTip(f.tooltip)
            row = widget
            if f.unit:
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(widget, 1)
                unit = QLabel(f.unit)
                unit.setMinimumWidth(52)
                h.addWidget(unit)
            label = QLabel(f.label)
            if f.tooltip:
                label.setToolTip(f.tooltip)
            self.layout_.addRow(label, row)

    def _browse(self, key, line, file_filter):
        start = line.text() or QSettings("MembraneNeuronSimulator", "App").value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Choose file", start, file_filter + ";;All files (*)")
        if path:
            line.setText(path)
            line.setToolTip(path)
            self.edited.emit(key, path)

    def _float_edited(self, key, widget):
        text = widget.text().replace(",", ".")
        try:
            self.edited.emit(key, float(text))
        except ValueError:
            pass


# -----------------------------
# Part properties
# -----------------------------

class PropertyPanel(QWidget):
    role_changed = Signal(list, str)
    props_changed = Signal(list, str, object)  # indices, key, value
    segment_highlight = Signal(list, object)   # indices, segment (1..N) or None

    def __init__(self, parent=None, roles=ROLES, role_fields=ROLE_FIELDS):
        super().__init__(parent)
        self.role_fields = role_fields
        layout = QVBoxLayout(self)
        self.title = QLabel("Select a part in the view or the model tree.")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(self.title)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: #555;")
        layout.addWidget(self.info)

        self.role_box = QGroupBox("Role")
        rl = QVBoxLayout(self.role_box)
        self.role_combo = QComboBox()
        for role in roles:
            self.role_combo.addItem(role_icon(role), role)
        self.role_combo.activated.connect(self._role_activated)
        rl.addWidget(self.role_combo)
        self.role_help = QLabel()
        self.role_help.setWordWrap(True)
        self.role_help.setStyleSheet("color: #555;")
        rl.addWidget(self.role_help)
        layout.addWidget(self.role_box)

        self.props_box = QGroupBox("Properties")
        pl = QVBoxLayout(self.props_box)
        self.form = FieldForm()
        self.form.edited.connect(self._edited)
        self.form.highlight.connect(lambda k: self.segment_highlight.emit(self.indices, k))
        pl.addWidget(self.form)
        layout.addWidget(self.props_box)

        self.coupling_box = QGroupBox("Pressure acts on")
        cl = QVBoxLayout(self.coupling_box)
        self.coupling_label = QLabel()
        self.coupling_label.setWordWrap(True)
        cl.addWidget(self.coupling_label)
        layout.addWidget(self.coupling_box)
        layout.addStretch(1)

        self.indices, self.parts, self.bodies = [], [], []
        self.set_selection([], [], [])

    def set_selection(self, indices, parts, bodies, couplings_text=None):
        self.indices, self.parts, self.bodies = list(indices), parts, bodies
        has = bool(self.indices)
        self.role_box.setVisible(has)
        self.props_box.setVisible(False)
        self.coupling_box.setVisible(False)
        if not has:
            self.title.setText("Select a part in the view or the model tree.\nCtrl+click selects several.")
            self.info.setText("")
            return

        selected = [parts[i] for i in self.indices]
        if len(selected) == 1:
            body = bodies[self.indices[0]]
            self.title.setText(selected[0].name)
            size = " × ".join(f"{d:.4g}" for d in body.size)
            self.info.setText(f"Volume {body.volume:.6g} mm³ · bounding box {size} mm · "
                              f"thin-body thickness ≈ {body.thickness_estimate:.3g} mm")
        else:
            self.title.setText(f"{len(selected)} parts selected")
            self.info.setText(", ".join(p.name for p in selected))

        roles = {p.role for p in selected}
        role = roles.pop() if len(roles) == 1 else None
        self.role_combo.blockSignals(True)
        if role is None:
            self.role_combo.setCurrentIndex(-1)
            self.role_help.setText("Mixed roles - choose one to assign it to all selected parts.")
        else:
            self.role_combo.setCurrentText(role)
            self.role_help.setText(ROLE_HELP[role])
        self.role_combo.blockSignals(False)

        if role is not None and self.role_fields[role]:
            self.props_box.setVisible(True)
            self.form.build(self.role_fields[role], selected[0].props,
                            {"__role__": role, "__chambers__": [p.name for p in parts if p.role == CHAMBER]})
        if role == CHAMBER and couplings_text is not None:
            self.coupling_box.setVisible(True)
            self.coupling_label.setText(couplings_text)

    def _role_activated(self, _):
        if self.indices:
            self.role_changed.emit(self.indices, self.role_combo.currentText())

    def _edited(self, key, value):
        if self.indices:
            self.props_changed.emit(self.indices, key, value)


# -----------------------------
# Solver settings
# -----------------------------

class SolverPanel(QWidget):
    changed = Signal(str, object)

    NOTE = ("All chamber pressures are ramped from zero to their set values over the load "
            "steps; steps are subdivided automatically when Newton does not converge.\n\n"
            "Units: mm, N, MPa (pressures entered in kPa).")

    def __init__(self, parent=None, title="Static Newton-Raphson solver", fields=SOLVER_FIELDS, note=None):
        super().__init__(parent)
        self.fields = fields
        layout = QVBoxLayout(self)
        box = QGroupBox(title)
        bl = QVBoxLayout(box)
        self.form = FieldForm()
        self.form.edited.connect(self.changed.emit)
        bl.addWidget(self.form)
        note = QLabel(self.NOTE if note is None else note)
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        bl.addWidget(note)
        layout.addWidget(box)
        layout.addStretch(1)

    def load(self, settings):
        self.form.build(self.fields, vars(settings))


# -----------------------------
# Results
# -----------------------------

class ResultsPanel(QWidget):
    display_changed = Signal()
    export_vtk = Signal()
    export_csv = Signal()
    screenshot = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.status = QLabel("No results yet. Press Solve (F5).")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        display = QGroupBox("Display")
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
        step_row = QWidget()
        sl = QHBoxLayout(step_row)
        sl.setContentsMargins(0, 0, 0, 0)
        self.step = QSlider(Qt.Horizontal)
        self.step.valueChanged.connect(self._step_changed)
        self.step_label = QLabel("")
        self.step_label.setMinimumWidth(60)
        sl.addWidget(self.step, 1)
        sl.addWidget(self.step_label)
        form.addRow("Load step", step_row)
        self.fixed_range = QCheckBox("Same colour range for all load steps")
        self.fixed_range.setChecked(True)
        self.fixed_range.toggled.connect(lambda _: self.display_changed.emit())
        form.addRow(self.fixed_range)
        layout.addWidget(display)

        chambers = QGroupBox("Chambers")
        cl = QVBoxLayout(chambers)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Chamber", "P [kPa]", "ΔV [mm³]", "V [mm³]"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        cl.addWidget(self.table)
        self.shell_label = QLabel()
        self.shell_label.setWordWrap(True)
        cl.addWidget(self.shell_label)
        layout.addWidget(chambers, 1)

        buttons = QHBoxLayout()
        for text, signal in (("Export VTK…", self.export_vtk), ("Export CSV…", self.export_csv),
                             ("Screenshot…", self.screenshot)):
            b = QPushButton(text)
            b.clicked.connect(signal.emit)
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self.history = []

    def set_results(self, history, status_text):
        self.history = history
        self.status.setText(status_text)
        self.step.blockSignals(True)
        self.step.setRange(0, max(0, len(history) - 1))
        self.step.setValue(len(history) - 1)
        self.step.blockSignals(False)
        self._update_step_label()

    def current_step(self):
        return self.history[self.step.value()] if self.history else None

    def _step_changed(self, _):
        self._update_step_label()
        self.display_changed.emit()

    def _update_step_label(self):
        step = self.current_step()
        if step and self.step.value() == 0:
            self.step_label.setText("0 (start)")
        else:
            self.step_label.setText(f"{step['load_factor']:.0%}" if step else "")

    def set_table(self, rows, shell_text=""):
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(value if isinstance(value, str) else fmt(value))
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
        self.shell_label.setText(shell_text)


def separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line
