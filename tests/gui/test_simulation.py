import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer

import sequoia.components.builtin  # noqa: F401
import sequoia.engines.outcome as outcome_module
from sequoia.engines.cache import Cancelled
from sequoia.engines.classical import simulate
from sequoia.engines.rates import simulate_rates
from sequoia.gui.document import SetupDocument
from sequoia.gui.results import ResultsDock, eng, ids_for_issue
from sequoia.gui.simulation import SimulationController
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-borankilic-Desktop-Research-Quantum-Technologies-SEQUOIA/"
    "5d0a0d99-6543-4fe7-aef3-c7336f569617/scratchpad"
)
WAIT = 60000


@pytest.fixture
def real_library():
    return Library(ROOT / "library")


def open_doc(name, real_library):
    return SetupDocument.open(ROOT / "setups" / f"{name}.yaml", real_library)


@pytest.fixture
def make_controller(qtbot):
    made = []

    def make(doc, **kwargs):
        ctrl = SimulationController(doc, **kwargs)
        made.append(ctrl)
        return ctrl

    yield make
    for ctrl in made:
        ctrl.shutdown()


class Counter:
    def __init__(self, ctrl):
        self.started = 0
        self.cancelled = 0
        self.finished = []
        self.failed = []
        self.busy = []
        ctrl.started.connect(lambda: setattr(self, "started", self.started + 1))
        ctrl.cancelled.connect(lambda: setattr(self, "cancelled", self.cancelled + 1))
        ctrl.finished.connect(self.finished.append)
        ctrl.failed.connect(self.failed.append)
        ctrl.busy_changed.connect(self.busy.append)


def test_classical_run_matches_engine(qtbot, make_controller, real_library):
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc)
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT) as blocker:
        ctrl.run_now()
    outcome = blocker.args[0]
    direct = simulate(doc.setup, real_library)
    assert outcome.ok
    assert outcome.rates is None
    assert outcome.revision == ctrl.revision
    assert outcome.elapsed > 0
    assert set(outcome.classical.links) == set(direct.links)
    for link_id, state in direct.links.items():
        assert outcome.classical.links[link_id].end.power == pytest.approx(state.end.power)
    assert outcome.classical.probes["p2"]["power_W"] == pytest.approx(
        direct.probes["p2"]["power_W"]
    )


def test_rate_run_matches_engine(qtbot, make_controller, real_library):
    doc = open_doc("hbt_heralded", real_library)
    ctrl = make_controller(doc)
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT) as blocker:
        ctrl.run_now()
    outcome = blocker.args[0]
    direct = simulate_rates(doc.setup, real_library)
    assert outcome.rates is not None
    assert set(outcome.rates.channels) == set(direct.channels)
    for name, entry in direct.channels.items():
        assert outcome.rates.channels[name]["singles_hz"] == pytest.approx(entry["singles_hz"])
    for name, entry in direct.analyses.items():
        assert outcome.rates.analyses[name]["type"] == entry["type"]
    assert {(i.code, i.subject) for i in outcome.issues} == {
        (i.code, i.subject) for i in direct.issues
    }


def test_debounce_coalesces_rapid_edits(qtbot, make_controller, real_library):
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc, debounce_ms=120)
    counts = Counter(ctrl)
    ctrl.start()
    qtbot.waitUntil(lambda: counts.started == 1, timeout=WAIT)
    qtbot.waitUntil(lambda: len(counts.finished) == 1, timeout=WAIT)
    counts.started = 0
    for i in range(6):
        doc.set_link_field("a1", "length", f"{101 + i} mm")
    qtbot.waitUntil(lambda: len(counts.finished) == 2, timeout=WAIT)
    qtbot.wait(300)
    assert counts.started == 1
    assert len(counts.finished) == 2
    assert counts.finished[-1].revision == ctrl.revision
    assert counts.busy[0] is True and counts.busy[-1] is False
    assert not ctrl.busy


def test_auto_run_off_waits_for_run_now(qtbot, make_controller, real_library):
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc, debounce_ms=20, auto_run=False)
    counts = Counter(ctrl)
    ctrl.start()
    doc.set_link_field("a1", "length", "150 mm")
    qtbot.wait(150)
    assert counts.started == 0
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT):
        ctrl.run_now()
    assert counts.started == 1


@pytest.fixture
def slow_first_call(monkeypatch):
    state = {"calls": 0, "entered": threading.Event(), "cancelled": 0}
    real = simulate

    def fake(setup, library, *, should_cancel=None, cache=None, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            state["entered"].set()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if should_cancel is not None and should_cancel():
                    state["cancelled"] += 1
                    raise Cancelled("cancelled")
                time.sleep(0.01)
        return real(setup, library, should_cancel=should_cancel, cache=cache, **kwargs)

    monkeypatch.setattr(outcome_module, "simulate", fake)
    return state


def test_newer_edit_cancels_running_run(qtbot, make_controller, real_library, slow_first_call):
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc, debounce_ms=10)
    counts = Counter(ctrl)
    ctrl.start()
    qtbot.waitUntil(slow_first_call["entered"].is_set, timeout=WAIT)
    doc.set_link_field("a1", "length", "130 mm")
    qtbot.waitUntil(lambda: len(counts.finished) == 1, timeout=WAIT)
    assert slow_first_call["cancelled"] == 1
    assert counts.cancelled == 1
    assert counts.started == 2
    assert counts.finished[0].revision == ctrl.revision
    assert counts.finished[0].classical.links["a1"].length == pytest.approx(0.130)


def test_stale_result_is_dropped(qtbot, make_controller, real_library, monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    calls = []
    real = simulate

    def fake(setup, library, *, should_cancel=None, cache=None, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            release.wait(20)
        return real(setup, library, cache=cache, **kwargs)

    monkeypatch.setattr(outcome_module, "simulate", fake)
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc, debounce_ms=10)
    counts = Counter(ctrl)
    ctrl.start()
    qtbot.waitUntil(entered.is_set, timeout=WAIT)
    doc.set_link_field("a1", "length", "140 mm")
    qtbot.wait(100)
    release.set()
    qtbot.waitUntil(lambda: len(counts.finished) == 1, timeout=WAIT)
    assert counts.cancelled == 1
    assert counts.finished[0].revision == ctrl.revision
    assert counts.finished[0].classical.links["a1"].length == pytest.approx(0.140)


def test_invalid_setup_yields_issues_not_exception(qtbot, make_controller, real_library):
    doc = open_doc("mzi", real_library)
    doc.set_link_field("o1", "to", "nowhere.in")
    ctrl = make_controller(doc)
    counts = Counter(ctrl)
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT) as blocker:
        ctrl.run_now()
    outcome = blocker.args[0]
    assert outcome.classical is None and outcome.rates is None
    assert outcome.has_errors and not outcome.ok
    assert any("nowhere" in i.subject + i.message for i in outcome.issues)
    assert counts.failed == []


def test_worker_exception_reports_failed(qtbot, make_controller, real_library, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(outcome_module, "simulate", boom)
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc)
    with qtbot.waitSignal(ctrl.failed, timeout=WAIT) as blocker:
        ctrl.run_now()
    assert "RuntimeError" in blocker.args[0] and "engine exploded" in blocker.args[0]
    assert "boom" in ctrl.last_traceback
    qtbot.waitUntil(lambda: not ctrl.busy, timeout=WAIT)


def test_ui_thread_stays_responsive(qtbot, make_controller, real_library, monkeypatch):
    real = simulate

    def slow(setup, library, **kwargs):
        time.sleep(0.6)
        return real(setup, library, **kwargs)

    monkeypatch.setattr(outcome_module, "simulate", slow)
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc)
    ticks = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT):
        ctrl.run_now()
    timer.stop()
    assert len(ticks) >= 10


def test_cache_is_reused_between_runs(qtbot, make_controller, real_library):
    doc = open_doc("laser_fiber_coupling", real_library)
    ctrl = make_controller(doc)
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT) as first:
        ctrl.run_now()
    cache = ctrl.cache
    misses = cache.part_misses
    assert misses > 0
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT) as second:
        ctrl.run_now()
    assert ctrl.cache is cache
    assert cache.part_misses == misses
    assert cache.part_hits > 0
    assert second.args[0].elapsed <= first.args[0].elapsed * 1.5


def test_stop_disconnects(qtbot, make_controller, real_library):
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc, debounce_ms=10)
    counts = Counter(ctrl)
    ctrl.start()
    qtbot.waitUntil(lambda: len(counts.finished) == 1, timeout=WAIT)
    ctrl.stop()
    doc.set_link_field("a1", "length", "160 mm")
    qtbot.wait(150)
    assert counts.started == 1


def test_ids_for_issue():
    assert ids_for_issue("'bs1'") == ("bs1",)
    assert ids_for_issue("link 'l3'") == ("l3",)
    assert ids_for_issue("port 'det.out'") == ("det",)
    assert ids_for_issue("probe 'pm'") == ("pm",)
    assert ids_for_issue("instance 'x'") == ("x",)
    assert ids_for_issue("data") == ()
    assert ids_for_issue("setup") == ()


def test_eng_format():
    assert eng(1.5e-3, "W") == "1.5 mW"
    assert eng(2.5e6, "Hz") == "2.5 MHz"
    assert eng(0, "s") == "0 s"
    assert eng(float("inf"), "Hz") == "inf Hz"
    assert eng(3e-9, "s") == "3 ns"


def run_outcome(qtbot, ctrl):
    with qtbot.waitSignal(ctrl.finished, timeout=WAIT) as blocker:
        ctrl.run_now()
    return blocker.args[0]


def test_results_dock_tabs(qtbot, make_controller, real_library):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    doc = open_doc("hbt_heralded", real_library)
    ctrl = make_controller(doc)
    dock = ResultsDock()
    qtbot.addWidget(dock)
    dock.connect_controller(ctrl)
    activated = []
    dock.issue_activated.connect(activated.append)
    outcome = run_outcome(qtbot, ctrl)
    assert dock.links.rowCount() == len(outcome.classical.links)
    assert dock.links.columnCount() > 8
    assert dock.channels.rowCount() == len(outcome.rates.detectors)
    assert dock.analyses.topLevelItemCount() >= 1
    assert "Done" in dock.status.text()
    dock.show()
    for index, name in enumerate(["diag", "links", "readouts", "counts"]):
        dock.tabs.setCurrentIndex(index)
        qtbot.wait(20)
        assert dock.grab().save(str(SCRATCH / f"results_hbt_{name}.png"))
    if dock.diagnostics.rowCount():
        dock.diagnostics.selectRow(0)
        assert len(activated) == 1
        dock.diagnostics.cellDoubleClicked.emit(0, 0)
        assert len(activated) == 2
    dock.clear()
    assert dock.links.rowCount() == 0 and dock.diagnostics.rowCount() == 0


def test_results_dock_invalid_and_failed(qtbot, make_controller, real_library):
    doc = open_doc("mzi", real_library)
    doc.set_link_field("o1", "to", "nowhere.in")
    ctrl = make_controller(doc)
    dock = ResultsDock()
    qtbot.addWidget(dock)
    dock.set_outcome(run_outcome(qtbot, ctrl))
    assert dock.diagnostics.rowCount() >= 1
    assert "Invalid" in dock.status.text()
    dock.set_failed("RuntimeError: nope")
    assert "nope" in dock.status.text()
    dock.show()
    assert dock.grab().save(str(SCRATCH / "results_invalid.png"))


def test_results_dock_100_links_is_fast(qtbot, make_controller, real_library):
    doc = open_doc("mzi", real_library)
    ctrl = make_controller(doc)
    outcome = run_outcome(qtbot, ctrl)
    template = next(iter(outcome.classical.links.values()))
    import dataclasses

    for i in range(100):
        key = f"x{i}"
        outcome.classical.links[key] = dataclasses.replace(template, id=key)
    dock = ResultsDock()
    qtbot.addWidget(dock)
    start = time.perf_counter()
    dock.set_outcome(outcome)
    assert time.perf_counter() - start < 1.0
    assert dock.readouts.topLevelItemCount() >= 1
    from sequoia.gui.results import _sort_by_column

    _sort_by_column(dock.links, 3)
    values = [
        float(dock.links.item(r, 3).text())
        for r in range(dock.links.rowCount())
        if dock.links.item(r, 3).text() not in ("no light", "")
    ]
    assert values == sorted(values)
    assert dock.links.rowCount() == len(outcome.classical.links)
