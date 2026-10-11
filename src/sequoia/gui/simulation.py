import threading
import time
import traceback
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from sequoia.engines.cache import Cancelled, RunCache
from sequoia.engines.classical import SimulationResult, simulate
from sequoia.engines.rates import RateResult, has_rate_devices, simulate_rates
from sequoia.errors import Issue, SequoiaError
from sequoia.gui.document import SetupDocument
from sequoia.library.store import Library
from sequoia.setup.model import Setup
from sequoia.setup.validate import SetupValidationError

DEFAULT_DEBOUNCE_MS = 150


@dataclass
class SimulationOutcome:
    revision: int
    issues: list[Issue] = field(default_factory=list)
    classical: SimulationResult | None = None
    rates: RateResult | None = None
    elapsed: float = 0.0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def ok(self) -> bool:
        return self.classical is not None and not self.has_errors


def compute_outcome(
    setup: Setup,
    library: Library,
    cache: RunCache | None,
    should_cancel,
    revision: int = 0,
) -> SimulationOutcome:
    """Validate and simulate; user-level problems become error issues, never exceptions.

    Raises ``Cancelled`` when ``should_cancel`` fires and re-raises unexpected exceptions.
    """
    started = time.perf_counter()

    def outcome(**kwargs) -> SimulationOutcome:
        return SimulationOutcome(revision=revision, elapsed=time.perf_counter() - started, **kwargs)

    try:
        classical = simulate(setup, library, should_cancel=should_cancel, cache=cache)
    except SetupValidationError as exc:
        return outcome(issues=list(exc.issues))
    except Cancelled:
        raise
    except SequoiaError as exc:
        return outcome(issues=[Issue("error", "simulation_failed", "setup", str(exc))])
    issues = list(classical.issues)
    rates = None
    try:
        if has_rate_devices(setup, library, cache, classical.network):
            rates = simulate_rates(
                setup, library, classical=classical, should_cancel=should_cancel, cache=cache
            )
            issues = list(rates.issues)
    except Cancelled:
        raise
    except SequoiaError as exc:
        issues.append(Issue("error", "rates_failed", "setup", str(exc)))
    return outcome(issues=issues, classical=classical, rates=rates)


class _RunSignals(QObject):
    done = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()


class _Run(QRunnable):
    def __init__(self, setup, library, cache, revision) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.setup = setup
        self.library = library
        self.cache = cache
        self.revision = revision
        self.cancel = threading.Event()
        self.signals = _RunSignals()

    def run(self) -> None:
        if self.cancel.is_set():
            self.signals.cancelled.emit()
            return
        try:
            result = compute_outcome(
                self.setup, self.library, self.cache, self.cancel.is_set, self.revision
            )
        except Cancelled:
            self.signals.cancelled.emit()
        except BaseException as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            self.signals.done.emit(result)


class SimulationController(QObject):
    """Debounced background simulation of a ``SetupDocument``.

    Each request snapshots the (immutable-by-convention) ``Setup`` and runs ``simulate`` and, for
    setups with rate devices, ``simulate_rates`` on a single-thread pool, so at most one engine
    runs at a time and the per-document ``RunCache`` is only used from that worker. The cache
    wraps the document's ``Library`` (``CachedLibrary``); the library object itself is never
    mutated and holds only its root path. A newer request sets the cancellation flag of every
    older run; a run whose revision is no longer the document's is dropped and reported as
    ``cancelled``.
    """

    started = Signal()
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    busy_changed = Signal(bool)

    def __init__(
        self,
        document: SetupDocument,
        parent: QObject | None = None,
        *,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        auto_run: bool = True,
    ) -> None:
        super().__init__(parent)
        self.document = document
        self.auto_run = auto_run
        self.cache = RunCache()
        self._cache_library: Library | None = None
        self.last_outcome: SimulationOutcome | None = None
        self.last_traceback = ""
        self._revision = 0
        self._running = False
        self._active: list[_Run] = []
        self._busy = False
        self._connected = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._launch)

    @property
    def debounce_ms(self) -> int:
        return self._timer.interval()

    @debounce_ms.setter
    def debounce_ms(self, value: int) -> None:
        self._timer.setInterval(int(value))

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self) -> None:
        if not self._connected:
            self.document.changed.connect(self._on_changed)
            self._connected = True
        if self.auto_run:
            self._request(immediate=False)

    def stop(self) -> None:
        if self._connected:
            self.document.changed.disconnect(self._on_changed)
            self._connected = False
        self._timer.stop()
        for run in self._active:
            run.cancel.set()
        self._update_busy()

    def shutdown(self, timeout_ms: int = 5000) -> None:
        self.stop()
        self._pool.waitForDone(timeout_ms)

    def run_now(self) -> None:
        self._request(immediate=True)

    def clear_cache(self) -> None:
        self.cache = RunCache()
        self._cache_library = None

    def _on_changed(self) -> None:
        self._revision += 1
        for run in self._active:
            run.cancel.set()
        if self.auto_run:
            self._timer.start()
        self._update_busy()

    def _request(self, immediate: bool) -> None:
        if immediate:
            self._revision += 1
            self._timer.stop()
            self._launch()
        else:
            self._revision += 1
            self._timer.start()
            self._update_busy()

    def _launch(self) -> None:
        for run in self._active:
            run.cancel.set()
        library = self.document.library
        if self._cache_library is not library:
            self.cache = RunCache()
            self._cache_library = library
        run = _Run(self.document.setup, library, self.cache, self._revision)
        run.signals.done.connect(lambda outcome, r=run: self._on_done(r, outcome))
        run.signals.failed.connect(lambda msg, tb, r=run: self._on_failed(r, msg, tb))
        run.signals.cancelled.connect(lambda r=run: self._on_cancelled(r))
        self._active.append(run)
        self._pool.start(run)
        self.started.emit()
        self._update_busy()

    def _retire(self, run: _Run) -> None:
        if run in self._active:
            self._active.remove(run)

    def _is_stale(self, run: _Run) -> bool:
        return run.cancel.is_set() or run.revision != self._revision

    def _on_done(self, run: _Run, outcome: SimulationOutcome) -> None:
        self._retire(run)
        if self._is_stale(run):
            self.cancelled.emit()
        else:
            self.last_outcome = outcome
            self.finished.emit(outcome)
        self._update_busy()

    def _on_failed(self, run: _Run, message: str, trace: str) -> None:
        self._retire(run)
        if self._is_stale(run):
            self.cancelled.emit()
        else:
            self.last_traceback = trace
            self.failed.emit(message)
        self._update_busy()

    def _on_cancelled(self, run: _Run) -> None:
        self._retire(run)
        self.cancelled.emit()
        self._update_busy()

    def _update_busy(self) -> None:
        busy = self._timer.isActive() or bool(self._active)
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)
