from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from sequoia.fock import Circuit
from sequoia.fock.detectors import IdealPNR, PseudoPNR, Threshold
from sequoia.fock.engine import Limits
from sequoia.fock.experiment import (
    DetectorSpec,
    Experiment,
    load_experiment,
    run_experiment,
    save_experiment,
)
from sequoia.gui.fock_panel import FockTab
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
WAIT = 120000


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(Library(ROOT / "library"), settings=settings)
    w.confirm_discard = lambda _w: "discard"
    w.show_error = lambda title, message: pytest.fail(f"{title}: {message}")
    qtbot.addWidget(w)
    w.show()
    return w


@pytest.fixture
def tab(qtbot):
    t = FockTab()
    t.confirm_replace = lambda _t: "discard"
    qtbot.addWidget(t)
    yield t
    t.shutdown()


def open_template(qtbot, w, name):
    assert w.open_path(ROOT / "templates" / f"{name}.yaml")
    qtbot.waitUntil(lambda: w.controller.last_outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not w.controller.busy, timeout=WAIT)
    qtbot.wait(30)


def settle(qtbot, t):
    qtbot.waitUntil(lambda: not t.runner.busy, timeout=WAIT)
    qtbot.wait(30)


def same_physics(a: Experiment, b: Experiment):
    ra, rb = run_experiment(a), run_experiment(b)
    assert set(ra.probabilities) == set(rb.probabilities)
    for s, p in ra.probabilities.items():
        assert rb.probabilities[s] == pytest.approx(p, abs=1e-12)
    assert set(ra.conditionals) == set(rb.conditionals)
    for k, c in ra.conditionals.items():
        assert rb.conditionals[k].success_probability == pytest.approx(c.success_probability)
        for s, p in c.probs.items():
            assert rb.conditionals[k].probs[s] == pytest.approx(p)


def rich_experiment() -> Experiment:
    c = Circuit(4).beam_splitter(0, 1).beam_splitter(2, 3, 1.0)
    dets = [
        DetectorSpec(
            (0,), PseudoPNR(wires=2, weights=(0.9, 0.1), wire_efficiencies=(0.5, 0.8), dark=0.2)
        ),
        DetectorSpec((1,), Threshold(efficiency=0.7, dark=0.05)),
        DetectorSpec((2, 3), IdealPNR(efficiency=0.9, dark=0.01)),
    ]
    return Experiment(
        c,
        [1, 1, 1, 0],
        tags=[[0], [0], ["x"], []],
        brightness=0.123456789,
        indistinguishability=0.987654321,
        detectors=dets,
        shots=0,
        seed=None,
        conditions={"herald": {0: 1}, "pair": {1: (">=", 1), 2: 1}},
        name="rich",
        description="preserve this",
        mode_labels=["a", "b", "c", "d"],
        limits=Limits(max_photons=12, max_work=1e9),
        rate_hz=1e6,
        duration_s=3.0,
    )


@pytest.mark.parametrize(
    "exp",
    [
        rich_experiment(),
        Experiment(
            Circuit(2), [0, 0], detectors=[DetectorSpec((0,), PseudoPNR(wires=4, dark=0.2))]
        ),
        Experiment(Circuit(1), [1], conditions={"keep": {0: 1}}, name="save me"),
    ],
)
def test_editor_round_trip_preserves_the_experiment(tab, exp, tmp_path):
    tab.set_experiment(exp)
    back = tab.experiment()
    assert back.to_dict() == exp.to_dict()
    same_physics(exp, back)
    path = tmp_path / "e.yaml"
    assert tab.save_path(path)
    assert tab.open_path(path)
    again = tab.experiment()
    assert again.to_dict() == load_experiment(path).to_dict()
    same_physics(exp, again)
    assert not tab.editor_dirty


def test_review_probe_detector_and_conditions_survive(tab):
    d = PseudoPNR(wires=2, weights=(0.9, 0.1), wire_efficiencies=(0.5, 0.8), dark=0.2)
    e = Experiment(
        Circuit(1),
        [1],
        detectors=[DetectorSpec((0,), d)],
        conditions={"herald": {0: 1}},
        description="preserve this",
    )
    tab.set_experiment(e)
    r = tab.experiment()
    assert r.detectors[0].model == d
    assert r.conditions == {"herald": {0: 1}}
    assert r.description == "preserve this"
    assert "custom wire" in tab.preserved_label.text()


def test_editing_one_field_keeps_the_others(tab):
    exp = rich_experiment()
    tab.set_experiment(exp)
    tab.input_table.item(1, 4).setText("0.5")
    back = tab.experiment()
    assert back.detectors[1].model == Threshold(efficiency=0.5, dark=0.05)
    assert back.detectors[0].model == exp.detectors[0].model
    assert back.conditions == exp.conditions and back.seed is None
    assert back.brightness == exp.brightness
    tab.input_table.item(0, 7).setText("0.3")
    model = tab.experiment().detectors[0].model
    assert model.dark == 0.3 and model.weights == (0.9, 0.1)
    tab.brightness.setValue(0.5)
    assert tab.experiment().brightness == 0.5
    assert tab.editor_dirty


def test_fractional_photon_numbers_are_rejected(tab):
    from sequoia.fock.circuit import FockError

    tab.set_experiment(Experiment(Circuit(2), [1, 1]))
    tab.input_table.item(0, 1).setText("1.5")
    with pytest.raises(FockError):
        tab.experiment()
    tab.input_table.item(0, 1).setText("1")
    tab.input_table.cellWidget(0, 3).setCurrentText("pseudo-PNR")
    tab.input_table.item(0, 5).setText("2.7")
    with pytest.raises(FockError):
        tab.experiment()


def test_extracted_fock_template_runs_identically_in_the_editor(window, qtbot):
    open_template(qtbot, window, "fock_hom")
    tab = window.results.fock
    tab.confirm_replace = lambda _t: "discard"
    tab.run()
    settle(qtbot, tab)
    setup = tab.setup_result
    assert setup is not None
    tab.extract_to_editor()
    settle(qtbot, tab)
    exp = tab.experiment()
    assert any(spec.model.dark > 0 for spec in exp.detectors)
    tab.run()
    settle(qtbot, tab)
    res = tab.result
    groups = [modes for modes, _ in exp.layout().groups]
    index = [groups.index(tuple(tab.extraction.detector_modes[d])) for d in setup.detector_ids]
    marginal: dict = {}
    for s, p in res.probabilities.items():
        key = tuple(s[i] for i in index)
        marginal[key] = marginal.get(key, 0.0) + p
    for s, p in setup.detected.items():
        assert marginal.get(s, 0.0) == pytest.approx(p, abs=1e-6)


def test_gaussian_result_finishing_after_an_edit_is_never_current(window, qtbot):
    open_template(qtbot, window, "bell_test")
    g = window.results.gaussian
    g.sweep.setChecked(False)
    g.run()
    window.document.set_setting("hwp_a", "angle", "10 deg")
    qtbot.waitUntil(lambda: g.outcome is not None and not g.runner.busy, timeout=WAIT)
    assert "outdated" in g.status.text()
    g.run()
    qtbot.waitUntil(lambda: not g.runner.busy, timeout=WAIT)
    qtbot.wait(30)
    assert "outdated" not in g.status.text()
    g.scale.setValue(2.0)
    assert "outdated" in g.status.text()


def test_gaussian_without_tagged_detectors_is_not_a_result(qtbot):
    from sequoia.engines.gaussian import GaussianResult
    from sequoia.gui.gaussian_panel import GaussianOutcome, GaussianTab

    g = GaussianTab(lambda: (None, None))
    qtbot.addWidget(g)
    g.begin_run()
    g._done(GaussianOutcome(GaussianResult([], 1.0, {}, {}, {})))
    assert g.outcome is None and "not computed" in g.status.text()
    g.runner.shutdown()


def test_fock_template_flags_its_acquisition_regime(window, qtbot):
    open_template(qtbot, window, "fock_hom")
    codes = {i.code for i in window.controller.last_outcome.issues}
    assert "pump_leakage" not in codes and "fock_source_mean_flux" in codes
    tab = window.results.fock
    tab.run()
    settle(qtbot, tab)
    assert "Acquisition regime not modelled" in tab.model_card.text()
    assert "dead time" in tab.model_card.text()
    header = tab.detector_table.horizontalHeaderItem(6).text()
    assert "Potential" in header and "dead time" in header
    g = window.results.gaussian
    g.sweep.setChecked(False)
    g.run()
    qtbot.waitUntil(lambda: not g.runner.busy, timeout=WAIT)
    qtbot.wait(30)
    assert g.outcome is None and "Fock-state source" in g.status.text()


def test_edit_marks_results_outdated_without_a_base_run(window, qtbot):
    open_template(qtbot, window, "bell_test")
    window.set_auto_run(False)
    r = window.results
    r.gaussian.sweep.setChecked(False)
    r.gaussian.run()
    qtbot.waitUntil(lambda: r.gaussian.outcome is not None, timeout=WAIT)
    assert "outdated" not in r.gaussian.status.text()
    window.document.set_setting("hwp_a", "angle", "10 deg")
    assert "outdated" in r.gaussian.status.text()


def test_monte_carlo_settings_and_schmidt_interval(window, qtbot):
    open_template(qtbot, window, "bell_test")
    mc = window.results.montecarlo
    mc.schmidt.setValue(0.5)
    mc.run()
    assert not mc.runner.busy and "not a mode number" in mc.status.text()
    mc.schmidt.setValue(0.0)
    mc.duration.setValue(0.02)
    mc.seed.setValue(3)
    mc.run()
    qtbot.waitUntil(lambda: mc.outcome is not None and not mc.runner.busy, timeout=WAIT)
    assert "outdated" not in mc.status.text()
    mc.seed.setValue(4)
    assert "outdated" in mc.status.text()


def test_library_edit_marks_results_outdated(window, qtbot):
    open_template(qtbot, window, "bell_test")
    g = window.results.gaussian
    g.sweep.setChecked(False)
    g.run()
    qtbot.waitUntil(lambda: g.outcome is not None, timeout=WAIT)
    window.set_library(window.library)
    assert "outdated" in g.status.text()


def test_editor_reset_and_refresh_cannot_resurrect_a_result(tab, qtbot):
    tab.mode_box.setCurrentIndex(1)
    tab.set_experiment(Experiment(Circuit(1), [1], rate_hz=100, duration_s=1))
    tab.run()
    settle(qtbot, tab)
    assert tab.result is not None and tab.results_table.rowCount() > 0
    tab.reset()
    tab._refresh_results()
    assert tab.result is None and tab.results_table.rowCount() == 0


def test_editor_result_is_marked_when_the_circuit_changes(tab, qtbot):
    tab.mode_box.setCurrentIndex(1)
    tab.set_experiment(Experiment(Circuit(2).beam_splitter(0, 1), [1, 1]))
    tab.run()
    settle(qtbot, tab)
    assert "outdated" not in tab.success_label.text()
    tab.input_table.item(0, 1).setText("2")
    assert "outdated" in tab.success_label.text()


def test_replacing_an_edited_circuit_asks_first(tab, tmp_path):
    path = tmp_path / "other.yaml"
    save_experiment(Experiment(Circuit(1), [1]), path)
    tab.set_experiment(Experiment(Circuit(2), [1, 1]))
    tab._saved_key = tab.editor_key()
    tab.input_table.item(0, 1).setText("2")
    asked = []
    tab.confirm_replace = lambda _t: asked.append(1) or "cancel"
    assert not tab.open_path(path)
    assert asked and tab.experiment().occupations == [2, 1]
    tab.confirm_replace = lambda _t: "discard"
    assert tab.open_path(path)
    assert tab.experiment().circuit.n_modes == 1


def test_zero_duration_is_allowed_in_the_photon_counts_tab(tab):
    tab.duration.setValue(0.0)
    assert tab.duration.value() == 0.0


def test_sweep_candidates_follow_the_document_structure(window, qtbot):
    open_template(qtbot, window, "bell_test")
    sweeps = window.results.sweeps
    before = {c.target for c in sweeps.rows[0].candidates}
    window.document.remove_instance("hwp_a")
    after = {c.target for c in sweeps.rows[0].candidates}
    assert after != before
    assert not any(t.startswith("settings:hwp_a.") for t in after)
