"""
What the GUI's Study tab follows live. Every command appends events to <study>/.mns/events.jsonl (one JSON
object per line); running jobs also keep <study>/.mns/progress/<job>.json up to date and write the current
deformed shape to <study>/.mns/live/<design>.npz (both replaced atomically, at most every LIVE_PERIOD s).

Event kinds: command (a command started: its text and the agent's reason), done, error, note (the agent's
own message), design (a design was created or rebuilt), file (a figure, report or result was written),
progress (start/end of a long solve), frame (a new live shape).
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

LIVE_PERIOD = 1.0      # s between live shapes
PROGRESS_PERIOD = 0.5  # s between progress file updates


class EventLog:
    def __init__(self, study_root, job=None):
        self.root = Path(study_root)
        self.folder = self.root / ".mns"
        self.folder.mkdir(exist_ok=True)
        self.path = self.folder / "events.jsonl"
        self.job = job
        self._last_progress = 0.0
        self._last_frame = 0.0

    def emit(self, kind, text="", **data):
        event = {"t": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, "text": text}
        if self.job:
            event["job"] = self.job
        event.update({k: v for k, v in data.items() if v is not None})
        line = json.dumps(event, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return event

    # -----------------------------

    def progress(self, design, fraction, text, force=False):
        """The running job's progress (fraction in 0..1, -1 unknown), for the GUI's progress bar."""
        now = time.time()
        if not force and now - self._last_progress < PROGRESS_PERIOD:
            return
        self._last_progress = now
        folder = self.folder / "progress"
        folder.mkdir(exist_ok=True)
        _atomic_json(folder / f"{self.job or 'foreground'}.json",
                     {"t": datetime.now().isoformat(timespec="milliseconds"), "design": design,
                      "fraction": float(fraction), "text": text, "pid": os.getpid()})

    def clear_progress(self):
        path = self.folder / "progress" / f"{self.job or 'foreground'}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def frame(self, design, bodies, label="", force=False):
        """The current deformed shape: bodies = [(name, kind, faces, x, scalar per node or None)]."""
        now = time.time()
        if not force and now - self._last_frame < LIVE_PERIOD:
            return
        self._last_frame = now
        folder = self.folder / "live"
        folder.mkdir(exist_ok=True)
        arrays = {"names": np.array([b[0] for b in bodies]), "kinds": np.array([b[1] for b in bodies]),
                  "label": np.array(label)}
        for k, (_, _, faces, x, u) in enumerate(bodies):
            arrays[f"faces{k}"] = np.asarray(faces, np.int32)
            arrays[f"x{k}"] = np.asarray(x, np.float32)
            if u is not None:
                arrays[f"u{k}"] = np.asarray(u, np.float32)
        target = folder / f"{design}.npz"
        tmp = folder / f"{design}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, target)
        self.emit("frame", label, design=design)


def _atomic_json(path, data):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_events(study_root, start=0):
    """(events, new offset): the events appended since byte offset `start` (whole lines only)."""
    path = Path(study_root) / ".mns" / "events.jsonl"
    if not path.exists():
        return [], 0
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read()
    end = data.rfind(b"\n") + 1
    events = []
    for line in data[:end].splitlines():
        try:
            events.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            pass
    return events, start + end
