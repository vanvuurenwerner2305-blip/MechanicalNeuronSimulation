"""A stand-in for the GUI's Worker thread: the simulator's job functions (run_sweep, run_grid, ...) report through
worker.check() / report() / log() / item.emit(); here those go to the study's event log, the progress file and a
list of rows."""
import time


class _Signal:
    def __init__(self, fn):
        self.emit = fn


class HeadlessWorker:
    def __init__(self, events, design_id, on_item=None, deadline=None):
        self.events = events
        self.design = design_id
        self.rows = []
        self.messages = []
        self._on_item = on_item
        self.deadline = deadline
        self.item = _Signal(self._item)

    def check(self):
        if self.deadline is not None and time.time() > self.deadline:
            raise TimeoutError("the run reached its time limit (--max-minutes)")

    def report(self, fraction, text=""):
        self.events.progress(self.design, fraction, text)

    def log(self, text):
        self.messages.append(text)

    def _item(self, row):
        self.rows.append(row)
        if self._on_item:
            self._on_item(row)
