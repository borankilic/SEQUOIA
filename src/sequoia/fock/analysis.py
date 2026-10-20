import math
import operator
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sequoia.fock.circuit import FockError
from sequoia.fock.engine import Distribution, State

_OPS = {
    "==": operator.eq,
    "=": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

Condition = Callable[[State], bool] | Mapping[int, Any]


@dataclass
class Conditional:
    """Result of conditioning: ``success_probability`` is the unnormalized mass of the accepted
    states; ``probs`` is their normalized distribution (empty when the success probability is 0)."""

    success_probability: float
    probs: dict[State, float]
    joint: dict[State, float]
    description: str = ""


def _entry_test(spec: Any) -> Callable[[int], bool]:
    if callable(spec):
        return spec
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        op, val = spec
        if op not in _OPS:
            raise FockError(f"unknown comparison {op!r}")
        return lambda n, f=_OPS[op], v=val: f(n, v)
    if isinstance(spec, (set, frozenset)):
        return lambda n, s=spec: n in s
    return lambda n, v=spec: n == v


def _predicate(condition: Condition) -> tuple[Callable[[State], bool], str]:
    if callable(condition):
        return condition, "predicate"
    tests = {int(m): _entry_test(spec) for m, spec in condition.items()}
    text = ", ".join(f"mode {m}: {condition[m]}" for m in condition)

    def pred(state: State) -> bool:
        return all(m < len(state) and t(state[m]) for m, t in tests.items())

    return pred, text


def condition(dist: Distribution | Mapping[State, float], cond: Condition) -> Conditional:
    probs = dist.probs if isinstance(dist, Distribution) else dist
    pred, text = _predicate(cond)
    joint = {s: p for s, p in probs.items() if pred(s)}
    success = float(sum(joint.values()))
    cond_p = {s: p / success for s, p in joint.items()} if success > 0.0 else {}
    return Conditional(success, cond_p, joint, text)


def marginal(dist: Distribution | Mapping[State, float], modes) -> dict[State, float]:
    probs = dist.probs if isinstance(dist, Distribution) else dist
    modes = [int(m) for m in modes]
    out: dict[State, float] = {}
    for s, p in probs.items():
        key = tuple(s[m] for m in modes)
        out[key] = out.get(key, 0.0) + p
    return dict(sorted(out.items()))


TAILS = ("unresolved", "condition", "error")


@dataclass
class Samples:
    """``counts`` holds the resolved outcomes. ``unresolved`` counts windows that fell in the
    probability mass the calculation did not resolve (truncated input weight), drawn as an
    explicit outcome; with ``tail='condition'`` the shots are conditioned on the resolved sector
    and ``conditioned_on`` is its probability."""

    shots: int
    seed: int | None
    counts: dict[State, int]
    sequence: list[State | None] = field(default_factory=list)
    unresolved: int = 0
    unresolved_probability: float = 0.0
    conditioned_on: float | None = None

    def frequencies(self) -> dict[State, float]:
        return {s: c / self.shots for s, c in self.counts.items()} if self.shots else {}


def sample(
    dist: Distribution | Mapping[State, float],
    shots: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    tail: str = "unresolved",
) -> Samples:
    probs = dist.probs if isinstance(dist, Distribution) else dist
    if tail not in TAILS:
        raise FockError(f"tail must be one of {TAILS}, got {tail!r}")
    if shots < 0:
        raise FockError("shots must not be negative")
    if not probs:
        raise FockError("cannot sample from an empty distribution")
    states = sorted(probs)
    p = np.array([probs[s] for s in states], dtype=float)
    total = p.sum()
    if not total > 0.0:
        raise FockError("distribution has no probability mass")
    missing = max(1.0 - float(total), 0.0)
    if missing <= 1e-12:
        missing = 0.0
    if missing and tail == "error":
        raise FockError(
            f"{missing:.3g} of the probability is not resolved (truncated input); sample with "
            "tail='unresolved' or tail='condition', or raise the truncation"
        )
    rng = rng if rng is not None else np.random.default_rng(seed)
    if missing and tail == "unresolved":
        weights = np.append(p, missing)
        idx = rng.choice(len(weights), size=shots, p=weights / weights.sum())
        sequence = [states[i] if i < len(states) else None for i in idx]
    else:
        idx = rng.choice(len(states), size=shots, p=p / total)
        sequence = [states[i] for i in idx]
    counts = Counter(s for s in sequence if s is not None)
    unresolved = shots - sum(counts.values())
    return Samples(
        shots,
        seed,
        dict(sorted(counts.items())),
        sequence,
        unresolved,
        missing if tail == "unresolved" else 0.0,
        float(total) if missing and tail == "condition" else None,
    )


def format_state(state: State) -> str:
    return "|" + ",".join(str(n) for n in state) + ">"


def parse_state(text: str) -> State:
    t = text.strip().strip("|>").strip()
    try:
        return tuple(int(x) for x in t.replace(" ", "").split(",") if x != "")
    except ValueError as exc:
        raise FockError(f"cannot read {text!r} as a photon-number state") from exc


ACQUISITIONS = ("fixed_windows", "poisson")


@dataclass
class ExpectedCounts:
    """Expected events from per-window probabilities. ``per_window`` is the probability of each
    outcome in one detection window; ``per_second = per_window * windows_per_second``; ``total``
    is over ``duration_s``. ``sigma`` is the predicted shot-to-shot spread of the count:
    ``acquisition='fixed_windows'`` (default) treats the ``n_windows`` windows as independent
    trials, so counts are multinomial, ``Var N_i = N p_i (1 - p_i)``, ``Cov(N_i, N_j) = -N p_i
    p_j``; ``'poisson'`` is the rare-event approximation ``Var N_i = N p_i``, covariances zero."""

    per_window: dict[State, float]
    windows_per_second: float | None = None
    duration_s: float | None = None
    per_second: dict[State, float] | None = None
    total: dict[State, float] | None = None
    sigma: dict[State, float] | None = None
    n_windows: float | None = None
    acquisition: str = "fixed_windows"

    def covariance(self, a: State, b: State) -> float:
        if self.n_windows is None:
            raise FockError("covariances need a duration")
        pa, pb = self.per_window.get(a, 0.0), self.per_window.get(b, 0.0)
        if a == b:
            return count_variance(pa, self.n_windows, self.acquisition)
        return 0.0 if self.acquisition == "poisson" else -self.n_windows * pa * pb

    def sigma_sum(self, states) -> float:
        """Spread of the summed count of ``states`` (covariances included)."""
        if self.n_windows is None:
            raise FockError("uncertainties need a duration")
        p = float(sum(self.per_window.get(s, 0.0) for s in set(states)))
        return float(np.sqrt(count_variance(p, self.n_windows, self.acquisition)))


def count_variance(p: float, n_windows: float, acquisition: str = "fixed_windows") -> float:
    if acquisition not in ACQUISITIONS:
        raise FockError(f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}")
    if acquisition == "poisson":
        return n_windows * p
    return n_windows * p * max(1.0 - p, 0.0)


def _finite(name: str, value: float | None, positive: bool) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise FockError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(v):
        raise FockError(f"{name} must be finite, got {v}")
    if positive and v <= 0.0:
        raise FockError(f"{name} must be positive, got {v}")
    if not positive and v < 0.0:
        raise FockError(f"{name} must not be negative, got {v}")
    return v


def check_counts(
    rate_hz: float | None, window_s: float | None, duration_s: float | None
) -> tuple[float | None, float | None, float | None]:
    """Count settings: rate and window must be finite and positive; a duration must be finite
    and >= 0 (zero gives zero counts)."""
    return (
        _finite("rate_hz", rate_hz, True),
        _finite("window_s", window_s, True),
        _finite("duration_s", duration_s, False),
    )


def windows_per_second(rate_hz: float | None, window_s: float | None) -> float | None:
    rate_hz, window_s, _ = check_counts(rate_hz, window_s, None)
    if rate_hz is not None:
        return rate_hz
    if window_s is not None:
        return 1.0 / window_s
    return None


def expected_counts(
    dist: "Distribution | Conditional | Mapping[State, float]",
    rate_hz: float | None = None,
    window_s: float | None = None,
    duration_s: float | None = None,
    acquisition: str = "fixed_windows",
) -> ExpectedCounts:
    """``rate_hz`` is the number of windows per second (repetition rate); without it
    ``1 / window_s`` is used (back-to-back windows). A Conditional contributes its joint
    (unnormalised) probabilities, i.e. the rate of the accepted events."""
    if acquisition not in ACQUISITIONS:
        raise FockError(f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}")
    if isinstance(dist, Conditional):
        probs = dist.joint
    elif isinstance(dist, Distribution):
        probs = dist.probs
    else:
        probs = dist
    per_window = dict(probs)
    _, _, duration_s = check_counts(rate_hz, window_s, duration_s)
    wps = windows_per_second(rate_hz, window_s)
    if wps is None:
        if duration_s is not None:
            raise FockError("duration_s needs rate_hz or window_s")
        return ExpectedCounts(per_window, acquisition=acquisition)
    per_second = {s: p * wps for s, p in per_window.items()}
    if duration_s is None:
        return ExpectedCounts(per_window, wps, None, per_second, acquisition=acquisition)
    n_win = wps * duration_s
    total = {s: p * n_win for s, p in per_window.items()}
    sigma = {
        s: float(np.sqrt(count_variance(p, n_win, acquisition))) for s, p in per_window.items()
    }
    return ExpectedCounts(per_window, wps, duration_s, per_second, total, sigma, n_win, acquisition)


def outcome_table(
    dist: Distribution,
    detectors: Any = None,
    top: int | None = None,
    n_windows: float | None = None,
    acquisition: str = "fixed_windows",
) -> list[dict[str, Any]]:
    """Rows sorted by probability. ``detectors`` is a DetectorLayout applied first. Counts and
    their spread (see ExpectedCounts) use ``n_windows`` (default ``dist.info['n_windows']``)."""
    d = detectors.apply(dist) if detectors is not None else dist
    n_win = n_windows if n_windows is not None else dist.info.get("n_windows")
    rows = []
    for s, p in sorted(d.probs.items(), key=lambda kv: (-kv[1], kv[0])):
        row: dict[str, Any] = {"state": s, "label": format_state(s), "probability": p}
        if n_win is not None:
            row["counts"] = p * n_win
            row["sigma"] = float(np.sqrt(count_variance(p, n_win, acquisition)))
        rows.append(row)
    return rows[:top] if top is not None else rows
