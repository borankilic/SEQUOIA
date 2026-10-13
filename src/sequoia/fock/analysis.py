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


@dataclass
class Samples:
    shots: int
    seed: int | None
    counts: dict[State, int]
    sequence: list[State] = field(default_factory=list)

    def frequencies(self) -> dict[State, float]:
        return {s: c / self.shots for s, c in self.counts.items()} if self.shots else {}


def sample(
    dist: Distribution | Mapping[State, float],
    shots: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> Samples:
    probs = dist.probs if isinstance(dist, Distribution) else dist
    if shots < 0:
        raise FockError("shots must not be negative")
    if not probs:
        raise FockError("cannot sample from an empty distribution")
    states = sorted(probs)
    p = np.array([probs[s] for s in states], dtype=float)
    total = p.sum()
    if not total > 0.0:
        raise FockError("distribution has no probability mass")
    rng = rng if rng is not None else np.random.default_rng(seed)
    idx = rng.choice(len(states), size=shots, p=p / total)
    sequence = [states[i] for i in idx]
    counts = Counter(sequence)
    return Samples(shots, seed, dict(sorted(counts.items())), sequence)


def format_state(state: State) -> str:
    return "|" + ",".join(str(n) for n in state) + ">"


def parse_state(text: str) -> State:
    t = text.strip().strip("|>").strip()
    try:
        return tuple(int(x) for x in t.replace(" ", "").split(",") if x != "")
    except ValueError as exc:
        raise FockError(f"cannot read {text!r} as a photon-number state") from exc
