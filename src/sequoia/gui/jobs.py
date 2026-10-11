import gc
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from sequoia.engines.cache import Cancelled

_gc_lock = threading.Lock()
_gc_depth = 0
_gc_was_enabled = True


@contextmanager
def gc_paused() -> Iterator[None]:
    """Keep the cyclic garbage collector off while worker threads run.

    A collection that happens on a worker thread can finalise PySide wrappers of Qt objects
    (menus, actions, plot items); their destructors take Qt-internal locks and then need the
    GIL, which deadlocks against the GUI thread. Collection is done on the GUI thread when the
    job has finished instead.
    """
    global _gc_depth, _gc_was_enabled
    with _gc_lock:
        if _gc_depth == 0:
            _gc_was_enabled = gc.isenabled()
            gc.disable()
        _gc_depth += 1
    try:
        yield
    finally:
        with _gc_lock:
            _gc_depth -= 1
            if _gc_depth == 0 and _gc_was_enabled:
                gc.enable()


class _Signals(QObject):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class _Job(QRunnable):
    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.fn = fn
        self.cancel = threading.Event()
        self.signals = _Signals()

    def run(self) -> None:
        try:
            with gc_paused():
                result = self.fn(self.cancel.is_set, self.signals.progress.emit)
        except Cancelled:
            self.signals.cancelled.emit()
        except BaseException as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        else:
            self.signals.done.emit(result)


class JobRunner(QObject):
    """Runs one long calculation at a time on a worker thread.

    ``fn(should_cancel, progress)`` is called off the GUI thread; ``progress(done, total)`` may be
    called freely. Exactly one of ``done(result)``, ``failed(message)`` or ``cancelled()`` fires.
    """

    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._job: _Job | None = None

    @property
    def busy(self) -> bool:
        return self._job is not None

    def start(self, fn: Callable[..., Any]) -> bool:
        if self._job is not None:
            return False
        job = _Job(fn)
        job.signals.progress.connect(self.progress)
        job.signals.done.connect(lambda r, j=job: self._finish(j, self.done, r))
        job.signals.failed.connect(lambda m, j=job: self._finish(j, self.failed, m))
        job.signals.cancelled.connect(lambda j=job: self._finish(j, self.cancelled))
        self._job = job
        self.busy_changed.emit(True)
        self._pool.start(job)
        return True

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancel.set()

    def shutdown(self, timeout_ms: int = 5000) -> None:
        self.cancel()
        self._pool.waitForDone(timeout_ms)

    def _finish(self, job: _Job, signal, *args) -> None:
        if self._job is job:
            self._job = None
        gc.collect()
        signal.emit(*args)
        self.busy_changed.emit(self._job is not None)
