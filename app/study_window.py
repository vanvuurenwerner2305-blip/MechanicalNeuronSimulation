"""
Design study tab: a study folder worked on by the design agent (Claude Code, through the `mns` command line),
followed live. Nothing here controls the agent; the tab only reads the study folder:

  .mns/events.jsonl        what the agent ran and why, designs created, files written (polled)
  .mns/progress/*.json     progress of running solves
  .mns/live/<design>.npz   the deforming shape during a solve
  designs/<ID>/preview.npz the bodies of a built design, coloured by role

Buttons create a study (agent files, brief, the user's starting files imported as designs), open one, and start
the agent in Windows Terminal (start_agent.cmd in the study folder). A design opens in its own space for a full
look (neuron, activation function or full neuron tab).
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from pyvistaqt import QtInteractor
from qtpy.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QColor, QDesktopServices, QPixmap
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                            QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                            QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter,
                            QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

from .meshview import polydata

SIMULATOR = Path(__file__).resolve().parents[1]
POLL_MS = 700
KIND_COLORS = {"command": "#1c5cab", "done": "#2f7d32", "error": "#c62828", "note": "#6a1b9a", "design": "#ad5a00",
               "file": "#555555", "frame": "#999999"}


def _python():
    exe = Path(sys.executable)
    return str(exe.with_name("python.exe")) if exe.name.lower() == "pythonw.exe" else str(exe)


class NewStudyDialog(QDialog):
    """Folder, name, brief and starting files of a new study."""

    def __init__(self, parent=None, start_dir=""):
        super().__init__(parent)
        self.setWindowTitle("New design study")
        self.resize(640, 460)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.folder = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form.addRow("Study folder (new or empty)", row)
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        layout.addLayout(form)
        layout.addWidget(QLabel("<b>Brief</b> — what the agent should find out (goal, free parameters and ranges, "
                                "what to measure, budget). You can also edit brief.md later."))
        self.brief = QPlainTextEdit()
        template = SIMULATOR / "agent" / "brief_template.md"
        self.brief.setPlainText(template.read_text(encoding="utf-8") if template.exists() else "# Brief\n")
        layout.addWidget(self.brief, 1)
        layout.addWidget(QLabel("<b>Starting files</b> — your models (*.mns, *.mad, *.mfn); imported as designs."))
        self.files = QListWidget()
        self.files.setMaximumHeight(90)
        layout.addWidget(self.files)
        row = QHBoxLayout()
        add, remove = QPushButton("Add…"), QPushButton("Remove")
        add.clicked.connect(self._add)
        remove.clicked.connect(lambda: [self.files.takeItem(self.files.row(i)) for i in self.files.selectedItems()])
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.start_dir = start_dir

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Study folder", self.start_dir)
        if path:
            self.folder.setText(path)
            if not self.name.text():
                self.name.setText(Path(path).name)

    def _add(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Starting files", self.start_dir,
                                                "Models (*.mns *.mad *.mfn *.step *.stp)")
        for p in paths:
            self.files.addItem(p)

    def _accept(self):
        folder = Path(self.folder.text().strip()) if self.folder.text().strip() else None
        if folder is None:
            QMessageBox.warning(self, "New study", "Choose a folder for the study.")
            return
        if folder.exists() and any(folder.iterdir()):
            QMessageBox.warning(self, "New study", "The folder is not empty: choose a new or empty folder.")
            return
        self.accept()

    def values(self):
        return (self.folder.text().strip(), self.name.text().strip(), self.brief.toPlainText(),
                [self.files.item(k).text() for k in range(self.files.count())])


class _ViewportHandle:
    """What AppWindow expects of a space (viewport.close())."""

    def __init__(self, plotter):
        self.plotter = plotter

    def close(self):
        self.plotter.close()


class StudyWindow(QMainWindow):
    open_design = Signal(str)   # a .mns / .mad / .mfn path to open in its space

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Design study")
        self.root = None
        self.worker = None
        self.offset = 0
        self.events = []
        self.current = None          # design ID shown
        self.followed = None         # design of the latest event (follow mode)
        self._live_mtime = {}
        self._preview_mtime = {}
        self._designs_mtime = 0.0
        self.process = None
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(POLL_MS)
        last = QSettings("MembraneNeuronSimulator", "App").value("last_study", "")
        if last and (Path(last) / "study.json").exists():
            QTimer.singleShot(0, lambda: self.open_study(last))

    # -----------------------------
    # UI
    # -----------------------------

    def _build_ui(self):
        bar = QToolBar("Study")
        bar.setMovable(False)
        self.addToolBar(bar)
        for text, slot, tip in (("New study…", self.new_study, "Make a study folder: agent files, brief, your models"),
                                ("Open study…", self.choose_study, "Open an existing study folder"),
                                ("Start agent", self.start_agent, "Open Claude Code in Windows Terminal, in the study "
                                                                  "folder, with the mns command available"),
                                ("Open folder", self.open_folder, "Show the study folder in Explorer"),
                                ("Edit brief", self.edit_brief, "Open brief.md"),
                                ("Study report", self.open_report, "Open report/study.pdf")):
            action = bar.addAction(text, slot)
            action.setToolTip(tip)
        bar.addSeparator()
        self.follow = QCheckBox("Follow the agent")
        self.follow.setChecked(True)
        self.follow.setToolTip("Show the design the agent works on, and its shape while it solves.")
        bar.addWidget(self.follow)
        self.title = QLabel("  No study open.")
        bar.addWidget(self.title)

        # left: designs and the event log
        left = QSplitter(Qt.Vertical)
        self.designs = QTableWidget(0, 4)
        self.designs.setHorizontalHeaderLabels(["ID", "Name", "Space", "Status"])
        self.designs.verticalHeader().setVisible(False)
        self.designs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.designs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.designs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.designs.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.designs.itemSelectionChanged.connect(self._design_clicked)
        self.designs.doubleClicked.connect(lambda _: self._open_in_space())
        left.addWidget(self.designs)
        self.log = QListWidget()
        self.log.setWordWrap(True)
        self.log.itemClicked.connect(self._event_clicked)
        left.addWidget(self.log)
        left.setSizes([260, 520])

        # centre: 3D view with the current command above it
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(0, 0, 0, 0)
        self.now = QLabel("The agent's current step appears here.")
        self.now.setWordWrap(True)
        self.now.setTextFormat(Qt.RichText)
        self.now.setStyleSheet("padding: 6px; background: #f3f6fa; border-bottom: 1px solid #d0d7de;")
        cl.addWidget(self.now)
        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(16)
        self.progress.setVisible(False)
        self.progress_text = QLabel()
        row.addWidget(self.progress, 1)
        row.addWidget(self.progress_text, 2)
        cl.addLayout(row)
        self.plotter = QtInteractor(centre)
        self.plotter.set_background("#dfe5ec", top="#ffffff")
        self.plotter.add_axes(interactive=False)
        cl.addWidget(self.plotter.interactor, 1)
        self.view_label = QLabel()
        self.view_label.setStyleSheet("padding: 3px 6px; color: #555;")
        cl.addWidget(self.view_label)
        self.viewport = _ViewportHandle(self.plotter)

        # right: the design's pictures and files
        right = QWidget()
        rl = QVBoxLayout(right)
        self.design_title = QLabel("<b>No design selected</b>")
        self.design_title.setWordWrap(True)
        self.design_title.setTextFormat(Qt.RichText)
        rl.addWidget(self.design_title)
        row = QHBoxLayout()
        for text, slot in (("Open in its space", self._open_in_space), ("Folder", self._open_design_folder),
                           ("Report", self._open_design_report)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        rl.addLayout(row)
        self.images = QListWidget()
        self.images.setMaximumHeight(130)
        self.images.currentItemChanged.connect(self._show_image)
        rl.addWidget(self.images)
        self.picture = QLabel("Pictures and plots of the design appear here.")
        self.picture.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.picture.setMinimumWidth(380)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.picture)
        rl.addWidget(scroll, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(centre)
        split.addWidget(right)
        split.setSizes([380, 820, 460])
        self.setCentralWidget(split)

    # -----------------------------
    # Study
    # -----------------------------

    def _settings(self):
        return QSettings("MembraneNeuronSimulator", "App")

    def choose_study(self):
        path = QFileDialog.getExistingDirectory(self, "Open study folder", self._settings().value("last_dir", ""))
        if path:
            self.open_study(path)

    def open_study(self, path):
        path = Path(path)
        if not (path / "study.json").exists():
            QMessageBox.warning(self, "Open study", f"{path} is not a study folder (no study.json).")
            return
        self.root = path.resolve()
        self._settings().setValue("last_study", str(self.root))
        try:
            info = json.loads((self.root / "study.json").read_text(encoding="utf-8"))
        except ValueError:
            info = {}
        self.title.setText(f"  Study: <b>{info.get('name', self.root.name)}</b> — {self.root}")
        self.title.setTextFormat(Qt.RichText)
        self.offset, self.events, self.current, self.followed = 0, [], None, None
        self._live_mtime, self._preview_mtime = {}, {}
        self.log.clear()
        self.plotter.clear()
        self._refresh_designs(force=True)
        self.poll()

    def new_study(self):
        dialog = NewStudyDialog(self, self._settings().value("last_dir", ""))
        if dialog.exec() != QDialog.Accepted:
            return
        folder, name, brief, files = dialog.values()
        Path(folder).mkdir(parents=True, exist_ok=True)
        brief_path = Path(folder).parent / f".{Path(folder).name}_brief.md"
        brief_path.write_text(brief, encoding="utf-8")
        args = ["-m", "mns_api", "new-study", folder, "--brief", str(brief_path)] + (["--name", name] if name else [])
        for f in files:
            args += ["--from", f]
        self.now.setText("Creating the study and importing your models…")
        self.process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(SIMULATOR))
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(env)
        self.process.finished.connect(lambda *_: self._study_created(folder, brief_path))
        self.process.start(_python(), args)

    def _study_created(self, folder, brief_path):
        out = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace").strip().splitlines()
        try:
            brief_path.unlink()
        except OSError:
            pass
        try:
            result = json.loads(out[-1]) if out else {"ok": False, "error": "no output"}
        except ValueError:
            result = {"ok": False, "error": out[-1] if out else "no output"}
        if not result.get("ok"):
            QMessageBox.warning(self, "New study", f"Could not create the study:\n{result.get('error')}\n"
                                                   f"{result.get('hint', '')}")
            return
        if result.get("import_errors"):
            QMessageBox.warning(self, "New study", "Some files could not be imported:\n"
                                + "\n".join(result["import_errors"]))
        self.open_study(folder)
        self.now.setText("Study created. Check brief.md (Edit brief), then press <b>Start agent</b>.")

    def start_agent(self):
        if self.root is None:
            QMessageBox.information(self, "Start agent", "Open or create a study first.")
            return
        brief = self.root / "brief.md"
        template = SIMULATOR / "agent" / "brief_template.md"
        if brief.exists() and template.exists() and brief.read_text(encoding="utf-8") == \
                template.read_text(encoding="utf-8"):
            if QMessageBox.question(self, "Start agent", "brief.md is still the empty template. Start anyway?") \
                    != QMessageBox.Yes:
                return
        script = self.root / "start_agent.cmd"
        if not script.exists():
            from mns_api.workspace import write_launchers
            write_launchers(self.root)
        try:
            subprocess.Popen(["wt.exe", "-d", str(self.root), "--title", "Design agent", "cmd", "/k", str(script)])
        except OSError:
            subprocess.Popen(["cmd", "/c", "start", "Design agent", "cmd", "/k", str(script)], cwd=str(self.root))
        self.now.setText("Agent started in a terminal window. Its steps appear here as it works.")

    def open_folder(self):
        if self.root:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root)))

    def edit_brief(self):
        if self.root:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root / "brief.md")))

    def open_report(self):
        if self.root:
            pdf = self.root / "report" / "study.pdf"
            if pdf.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf)))
            else:
                QMessageBox.information(self, "Study report", "There is no study report yet.")

    # -----------------------------
    # Following the study
    # -----------------------------

    def poll(self):
        if self.root is None:
            return
        from mns_api.events import read_events
        try:
            new, self.offset = read_events(self.root, self.offset)
        except OSError:
            return
        refresh = False
        for event in new:
            self.events.append(event)
            if event.get("kind") != "frame":
                self._add_event(event)
            if event.get("design"):
                self.followed = event["design"]
            if event.get("kind") in ("design", "done", "error"):
                refresh = True
            if event.get("kind") == "command":
                why = f"<br><i>{event['why']}</i>" if event.get("why") else ""
                self.now.setText(f"<b>{_short_time(event)}</b> &nbsp; <code>{_html(event.get('text', ''))}</code>{why}")
            elif event.get("kind") in ("done", "error"):
                colour = KIND_COLORS[event["kind"]]
                self.now.setText(self.now.text().split("<br><span")[0]
                                 + f"<br><span style='color:{colour}'>{_html(event.get('text', ''))}</span>")
        self._refresh_designs(force=refresh)
        self._poll_progress()
        if self.follow.isChecked() and self.followed and self.followed != self.current:
            self.show_design(self.followed)
        self._poll_live()
        if new and self.current:
            self._fill_images(keep=True)

    def _add_event(self, event):
        kind = event.get("kind", "")
        text = event.get("text", "")
        design = event.get("design")
        line = f"{_short_time(event)}  {kind:<7} {design + '  ' if design else ''}{text}"
        if event.get("why"):
            line += f"\n    why: {event['why']}"
        item = QListWidgetItem(line)
        item.setForeground(QColor(KIND_COLORS.get(kind, "#333333")))
        item.setData(Qt.UserRole, design)
        self.log.addItem(item)
        if self.log.count() > 3000:
            self.log.takeItem(0)
        self.log.scrollToBottom()

    def _poll_progress(self):
        folder = self.root / ".mns" / "progress"
        latest = None
        if folder.exists():
            for p in folder.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                age = (datetime.now() - datetime.fromisoformat(data["t"])).total_seconds()
                if age < 120 and (latest is None or data["t"] > latest["t"]):
                    latest = data
        if latest is None:
            self.progress.setVisible(False)
            self.progress_text.setText("")
            return
        self.progress.setVisible(True)
        fraction = latest.get("fraction", -1)
        self.progress.setRange(0, 100 if fraction >= 0 else 0)
        if fraction >= 0:
            self.progress.setValue(int(100 * fraction))
        self.progress_text.setText(f"{latest.get('design', '')}: {latest.get('text', '')}")

    def _refresh_designs(self, force=False):
        folder = self.root / "designs"
        if not folder.exists():
            return
        try:
            mtime = max([folder.stat().st_mtime] + [p.stat().st_mtime for p in folder.glob("*/state.json")])
        except OSError:
            return
        if not force and mtime == self._designs_mtime:
            return
        self._designs_mtime = mtime
        rows = []
        for d in sorted(folder.iterdir()):
            if not d.is_dir() or len(d.name) < 5:
                continue
            state = {}
            try:
                state = json.loads((d / "state.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            rows.append((d.name[:4], d.name[5:], {"N": "neuron", "A": "activation", "F": "full"}.get(d.name[0], "?"),
                         state.get("status", "created")))
        self.designs.blockSignals(True)
        self.designs.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, v in enumerate(values):
                self.designs.setItem(r, c, QTableWidgetItem(v))
            if values[0] == self.current:
                self.designs.selectRow(r)
        self.designs.resizeColumnToContents(0)
        self.designs.blockSignals(False)

    def _design_folder(self, design_id):
        if self.root is None or not design_id:
            return None
        return next((d for d in (self.root / "designs").glob(f"{design_id}_*") if d.is_dir()), None)

    def _design_clicked(self):
        rows = self.designs.selectionModel().selectedRows()
        if rows:
            design = self.designs.item(rows[0].row(), 0).text()
            if design != self.current:
                self.follow.setChecked(False)  # the user looks around: stop following until ticked again
                self.show_design(design)

    def _event_clicked(self, item):
        design = item.data(Qt.UserRole)
        if design:
            self.follow.setChecked(False)
            self.show_design(design)

    def show_design(self, design_id):
        folder = self._design_folder(design_id)
        if folder is None:
            return
        self.current = design_id
        state = {}
        try:
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        why = ""
        try:
            import yaml
            why = (yaml.safe_load((folder / "design.yaml").read_text(encoding="utf-8")) or {}).get("why", "")
        except Exception:
            pass
        self.design_title.setText(f"<b>{folder.name}</b> — {state.get('status', 'created')}<br><i>{_html(why)}</i>")
        self._fill_images()
        self._show_preview(force=True)

    def _fill_images(self, keep=False):
        folder = self._design_folder(self.current)
        if folder is None:
            return
        current = self.images.currentItem().text() if keep and self.images.currentItem() else None
        files = sorted(list((folder / "renders").glob("*.png")) + list((folder / "results").glob("*.png")),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        names = [str(p.relative_to(folder)).replace("\\", "/") for p in files]
        if [self.images.item(k).text() for k in range(self.images.count())] == names:
            return
        self.images.blockSignals(True)
        self.images.clear()
        for n in names:
            self.images.addItem(n)
        self.images.blockSignals(False)
        pick = names.index(current) if current in names else 0
        if names:
            self.images.setCurrentRow(pick)
        else:
            self.picture.setText("No pictures yet.")

    def _show_image(self, item, _=None):
        folder = self._design_folder(self.current)
        if item is None or folder is None:
            return
        pixmap = QPixmap(str(folder / item.text()))
        if pixmap.isNull():
            return
        width = max(self.picture.parentWidget().width() - 20, 300)
        self.picture.setPixmap(pixmap.scaledToWidth(min(width, pixmap.width()), Qt.SmoothTransformation))

    # -----------------------------
    # 3D
    # -----------------------------

    def _load_preview(self, folder):
        path = folder / "preview.npz"
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            bodies = []
            for i, name in enumerate(data["names"]):
                if f"v{i}" in data:
                    bodies.append((str(name), str(data["roles"][i]), str(data["colors"][i]), float(data["opacity"][i]),
                                   data[f"v{i}"], data[f"f{i}"]))
            return bodies
        except Exception:
            return None

    def _show_preview(self, force=False, live=None):
        folder = self._design_folder(self.current)
        if folder is None:
            return
        preview = folder / "preview.npz"
        mtime = preview.stat().st_mtime if preview.exists() else 0
        if not force and live is None and self._preview_mtime.get(self.current) == mtime:
            return
        self._preview_mtime[self.current] = mtime
        bodies = self._load_preview(folder) or []
        self.plotter.clear()
        self.plotter.add_axes(interactive=False)
        live_names = set(live["names"]) if live else set()
        for name, role, color, opacity, v, f in bodies:
            if name in live_names:
                continue
            fluid = role in ("Fluid chamber", "Fluid")
            alpha = (0.1 if live else 0.25) if fluid else (0.3 if live else opacity)
            if role == "Rigid body":
                alpha = min(alpha, 0.4)
            self.plotter.add_mesh(polydata(v, f), color=color, opacity=alpha, smooth_shading=False, render=False)
        text = f"{self.current}: model"
        if live:
            values = np.concatenate([u for u in live["u"] if u is not None] or [np.zeros(1)])
            clim = (0.0, float(max(values.max(), 1e-9)))
            for k, name in enumerate(live["names"]):
                pd = polydata(live["x"][k], live["faces"][k])
                if live["u"][k] is not None:
                    pd.point_data["|u| [mm]"] = live["u"][k]
                    self.plotter.add_mesh(pd, scalars="|u| [mm]", cmap="turbo", clim=clim, render=False,
                                          show_scalar_bar=k == 0, smooth_shading=False)
                else:
                    self.plotter.add_mesh(pd, color="#e4572e", render=False)
            running = self.progress.isVisible() and self.progress_text.text().startswith(self.current)
            text = f"{self.current}: {'live' if running else 'last run'} — {live['label']}"
        if not getattr(self, "_camera_set", {}).get(self.current):
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self._camera_set = {self.current: True}
        self.plotter.render()
        self.view_label.setText(text)

    def _poll_live(self):
        if self.current is None or self.root is None:
            return
        path = self.root / ".mns" / "live" / f"{self.current}.npz"
        if not path.exists():
            self._show_preview()
            return
        mtime = path.stat().st_mtime
        if self._live_mtime.get(self.current) == mtime:
            self._show_preview()
            return
        self._live_mtime[self.current] = mtime
        try:
            data = np.load(path, allow_pickle=False)
            n = len(data["names"])
            live = {"names": [str(s) for s in data["names"]], "label": str(data["label"]),
                    "faces": [data[f"faces{k}"] for k in range(n)], "x": [data[f"x{k}"] for k in range(n)],
                    "u": [data[f"u{k}"] if f"u{k}" in data else None for k in range(n)]}
        except Exception:
            return
        self._show_preview(live=live)

    # -----------------------------
    # Opening a design
    # -----------------------------

    def _open_in_space(self):
        folder = self._design_folder(self.current)
        if folder is None:
            return
        for name in ("model.mns", "model.mad", "model.mfn"):
            if (folder / name).exists():
                self.open_design.emit(str(folder / name))
                return
        QMessageBox.information(self, "Open", "The design has not been built yet.")

    def _open_design_folder(self):
        folder = self._design_folder(self.current)
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _open_design_report(self):
        folder = self._design_folder(self.current)
        if folder and (folder / "report" / "design.pdf").exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder / "report" / "design.pdf")))
        else:
            QMessageBox.information(self, "Report", "This design has no report yet.")


def _short_time(event):
    return str(event.get("t", ""))[11:19]


def _html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
