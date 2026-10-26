import threading
from dataclasses import replace
from pathlib import Path

import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.engines.cache import Cancelled
from sequoia.fock import Circuit
from sequoia.fock import analysis as fa
from sequoia.fock.engine import Limits
from sequoia.fock.experiment import CANCELLED_SAMPLING, Experiment, run_experiment
from sequoia.fock.inputs import bell_pair
from sequoia.fock.preflight import MAX_PATTERNS, estimate_experiment
from sequoia.fock.run import SetupRequest, run_editor_experiment, run_setup_request
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")


def _circuit() -> Circuit:
    return Circuit(4).beam_splitter(0, 1).beam_splitter(2, 3).beam_splitter(1, 2, 0.7)


# Section 5: the split modules keep the public names


def test_public_names_still_importable():
    from sequoia.fock import from_setup
    from sequoia.gui import fock_panel

    for name in from_setup.__all__:
        assert hasattr(from_setup, name)
    for name in ("FockTab", "parse_filter", "ELEMENT_TYPES", "DETECTOR_CHOICES", "BAR_LIMIT"):
        assert hasattr(fock_panel, name)
    assert fock_panel.parse_filter("0=1, spad=2", ["x", "spad"]) == {0: ("=", 1), 1: ("=", 2)}


# U11: preflight estimate


@pytest.mark.parametrize(
    "exp",
    [
        Experiment(_circuit(), [1, 1, 1, 0]),
        Experiment(_circuit(), [1, 1, 1, 0], brightness=0.8, indistinguishability=0.9),
        Experiment(_circuit(), [2, 1, 0, 1], tags=[[0, 1], [0], [], [2]]),
        Experiment(Circuit(4).beam_splitter(0, 2).loss(1, 0.3), [1, 1, 0, 1]),
        Experiment(_circuit(), [0, 0, 0, 0], input_mixture=bell_pair(0, 1, 2, 3)),
    ],
)
def test_estimate_matches_the_engine_accounting(exp):
    est = estimate_experiment(exp)
    res = run_experiment(exp)
    assert est.ok
    assert est.work == pytest.approx(res.resources["estimated_work"])
    assert est.loss_modes == res.resources["loss_modes"]
    assert est.modes == exp.circuit.n_modes
    assert est.output_patterns >= len(res.raw.probs)
    assert est.photons == res.resources["photons"]
    assert "modes" in est.summary() and "output patterns" in est.summary()


def test_estimate_rejects_before_computing_with_a_remedy():
    many = Experiment(Circuit(3), [4, 4, 4], limits=Limits(max_photons=3))
    est = estimate_experiment(many)
    assert not est.ok and "photon limit" in est.errors[0] and "raise max photons" in est.errors[0]
    heavy = Experiment(
        Circuit(12).beam_splitter(0, 1), [1] * 10 + [0, 0], limits=Limits(max_work=1e3)
    )
    est = estimate_experiment(heavy)
    assert any("exceeds the limit" in e and "reduce modes" in e for e in est.errors)
    shots = estimate_experiment(Experiment(_circuit(), [1, 1, 0, 0], shots=10**13))
    assert any("shots" in e for e in shots.errors)
    assert estimate_experiment(Experiment(_circuit(), [1, 1, 0, 0], shots=10**8)).ok


def test_estimate_bounds_output_patterns():
    big = Experiment(Circuit(64).loss(0, 0.1), [1] * 16 + [0] * 48, limits=Limits(max_photons=16))
    est = estimate_experiment(big)
    assert est.output_patterns > MAX_PATTERNS
    assert any("output patterns" in e for e in est.errors)


def test_large_brightness_mixture_is_bounded_not_enumerated():
    exp = Experiment(Circuit(12), [1] * 12, brightness=0.9, limits=Limits(max_photons=12))
    est = estimate_experiment(exp)
    assert est.branches_bounded and est.branches == 3**12


# U11: cancellation during sampling, partial results labelled


def test_sequence_sampling_is_chunked_with_the_same_draws(monkeypatch):
    dist = {(1, 0): 0.3, (0, 1): 0.5, (1, 1): 0.1}
    whole = fa.sample(dist, 5000, seed=3)
    monkeypatch.setattr(fa, "SAMPLE_CHUNK", 777)
    chunked = fa.sample(dist, 5000, seed=3)
    assert chunked.sequence == whole.sequence and chunked.unresolved == whole.unresolved
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(Cancelled):
        fa.sample(dist, 5000, seed=3, should_cancel=cancel)
    with pytest.raises(Cancelled):
        fa.sample(dist, 5000, seed=3, sequence=False, should_cancel=lambda: True)


def _cancel_after(stage: int):
    flag = threading.Event()

    def progress(done, total):
        if done >= stage:
            flag.set()

    return flag.is_set, progress


def test_cancel_after_the_distribution_gives_a_labelled_partial_result():
    exp = Experiment(_circuit(), [1, 1, 0, 0], shots=1000, seed=1)
    should_cancel, progress = _cancel_after(2)
    res = run_experiment(exp, should_cancel, progress)
    assert res.status == "partial" and res.samples is None
    assert CANCELLED_SAMPLING in res.messages
    assert res.mass == pytest.approx(1.0)
    should_cancel, progress = _cancel_after(2)
    out = run_editor_experiment(exp, None, should_cancel, progress)
    assert out.status == "partial"
    assert run_experiment(exp).status == "complete"


def test_setup_run_cancelled_while_sampling_is_partial():
    setup = load_setup(ROOT / "templates" / "fock_hom.yaml")
    should_cancel, progress = _cancel_after(4)
    req = SetupRequest("run", setup, LIB, shots=100, seed=2)
    out = run_setup_request(req, should_cancel, progress)
    assert out.status == "partial" and out.result.status == "partial"
    assert out.result.samples is None and CANCELLED_SAMPLING in out.result.messages
    full = run_setup_request(req)
    assert full.status == "complete" and full.views is not None
    assert full.estimate.ok and full.estimate.modes == full.se.experiment.circuit.n_modes
    assert full.views.states[0][1] == max(full.result.detected.values())
    from sequoia.fock.records import run_record

    assert run_record(out.result)["status"] == "partial"


def test_setup_run_refused_by_preflight_keeps_inventory():
    setup = load_setup(ROOT / "templates" / "fock_hom.yaml")
    out = run_setup_request(SetupRequest("run", setup, LIB, max_photons=1))
    assert out.rejected and out.result is None
    assert out.error.startswith("not run:") and "Estimate:" in out.error
    assert out.components and out.inputs
    ok = run_setup_request(SetupRequest("import", setup, LIB))
    assert ok.estimate is not None and ok.estimate.ok and not ok.rejected


def test_setup_views_match_the_result():
    setup = load_setup(ROOT / "templates" / "fock_hom.yaml")
    out = run_setup_request(SetupRequest("run", setup, LIB, duration=2.0))
    res, v = out.result, out.views
    assert [s for s, _ in v.states] == sorted(res.detected, key=lambda s: -res.detected[s])
    for k, row in enumerate(v.detectors):
        p = sum(p for s, p in res.detected.items() if s[k] > 0)
        assert row.p_click == pytest.approx(p)
        assert row.total == pytest.approx(p * res.counts.windows_per_second * 2.0)
    assert sum(p for _, p, *_ in v.patterns) == pytest.approx(res.mass)
    assert sum(v.photon_numbers.values()) == pytest.approx(sum(res.photon_numbers.values()))
    assert replace(out.request, kind="import").kind == "import"
