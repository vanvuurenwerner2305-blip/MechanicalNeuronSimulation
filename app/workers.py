"""Background jobs (meshing, solving, sweeps) so the UI stays responsive."""
import traceback

from qtpy.QtCore import QThread, Signal


class Cancelled(Exception):
    pass


class Worker(QThread):
    """Runs fn(worker, *args) in a thread. fn reports through worker.report()/log() and calls
    worker.check() regularly so the job can be cancelled."""
    progress = Signal(float, str)   # fraction in [0, 1] (-1: unknown), message
    message = Signal(str)
    item = Signal(object)           # intermediate results (e.g. one sweep point)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self._fn, self._args = fn, args
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def check(self):
        if self._cancelled:
            raise Cancelled()

    def report(self, fraction, text=""):
        self.progress.emit(float(fraction), text)

    def log(self, text):
        self.message.emit(text)

    def run(self):
        try:
            self.succeeded.emit(self._fn(self, *self._args))
        except Cancelled:
            self.failed.emit("Cancelled.")
        except Exception as exc:  # shown to the user in the log
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
