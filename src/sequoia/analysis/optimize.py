from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize as sp

from sequoia.analysis.sweep import (
    SweepError,
    _parse_target,
    evaluate,
    get_value,
    metrics_of,
    with_values,
)
from sequoia.engines.cache import Cancelled, RunCache, check_cancel
from sequoia.errors import SequoiaError
from sequoia.library.store import Library
from sequoia.setup.model import Setup

METHODS = ("Powell", "Nelder-Mead", "L-BFGS-B", "differential_evolution")
PENALTY = 1e12


@dataclass(frozen=True)
class Variable:
    target: str
    lo: float
    hi: float
    unit: str = ""
    start: float | None = None


@dataclass
class OptimizeResult:
    values: dict[str, float]
    objective: float
    baseline: float
    evaluations: int
    success: bool
    message: str
    history: list[tuple[dict[str, float], float]] = field(default_factory=list)
    setup: Setup | None = None


def optimize(
    setup: Setup,
    library: Library,
    variables: Sequence[Variable],
    objective: str,
    *,
    maximize: bool = True,
    method: str = "Powell",
    max_evaluations: int = 300,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> OptimizeResult:
    """Tune parameters of ``setup`` to maximise (or minimise) one result metric.

    Variables are scaled to [0, 1] between their bounds, so unit magnitudes do not matter. The
    local methods start from ``Variable.start`` or the setup's current value (the bound midpoint
    when the parameter is not set explicitly); ``differential_evolution`` is global.
    """
    variables = list(variables)
    if not variables:
        raise SweepError("nothing to optimise: no variables")
    if method not in METHODS:
        raise SweepError(f"unknown method {method!r}; use one of {METHODS}")
    seen = set()
    for v in variables:
        _parse_target(v.target)
        if not (np.isfinite(v.lo) and np.isfinite(v.hi) and v.lo < v.hi):
            raise SweepError(f"{v.target}: bounds must be finite with lo < hi")
        if v.target in seen:
            raise SweepError(f"{v.target} appears twice")
        seen.add(v.target)
    cache = cache if cache is not None else RunCache()
    lo = np.array([v.lo for v in variables])
    hi = np.array([v.hi for v in variables])
    sign = -1.0 if maximize else 1.0
    history: list[tuple[dict[str, float], float]] = []

    def physical(z: np.ndarray) -> np.ndarray:
        return lo + np.clip(z, 0.0, 1.0) * (hi - lo)

    def score(x: np.ndarray) -> float:
        assignments = [(v.target, float(xi), v.unit) for v, xi in zip(variables, x, strict=True)]
        try:
            outcome = evaluate(
                with_values(setup, assignments),
                library,
                cache=cache,
                should_cancel=should_cancel,
            )
            value = metrics_of(outcome).get(objective) if outcome.classical else None
        except Cancelled:
            raise
        except SequoiaError:
            value = None
        if value is None or not np.isfinite(value):
            return PENALTY
        history.append(({v.target: float(xi) for v, xi in zip(variables, x, strict=True)}, value))
        if callback is not None:
            callback(len(history), value)
        return sign * value

    def fun(z: np.ndarray) -> float:
        check_cancel(should_cancel)
        return score(physical(np.asarray(z, dtype=float)))

    start = np.empty(len(variables))
    for i, v in enumerate(variables):
        current = v.start if v.start is not None else get_value(setup, v.target, v.unit)
        if current is None or not (v.lo <= current <= v.hi):
            current = 0.5 * (v.lo + v.hi)
        start[i] = current
    z0 = (start - lo) / (hi - lo)
    first = fun(z0)
    baseline = float("nan") if first == PENALTY else sign * first
    bounds = [(0.0, 1.0)] * len(variables)
    if method == "differential_evolution":
        res = sp.differential_evolution(
            fun,
            bounds,
            maxiter=max(1, max_evaluations // (15 * len(variables))),
            popsize=15,
            seed=0,
            polish=False,
            x0=z0,
        )
    elif method == "L-BFGS-B":
        res = sp.minimize(
            fun, z0, method=method, bounds=bounds, options={"maxfun": max_evaluations, "eps": 1e-3}
        )
    elif method == "Powell":
        res = sp.minimize(
            fun, z0, method=method, bounds=bounds, options={"maxfev": max_evaluations, "xtol": 1e-4}
        )
    else:
        res = sp.minimize(
            fun, z0, method=method, bounds=bounds, options={"maxfev": max_evaluations}
        )
    if history:
        best_values, best_objective = (max if maximize else min)(history, key=lambda h: h[1])
    else:
        best_values = {v.target: float(s) for v, s in zip(variables, start, strict=True)}
        best_objective = float("nan")
    assignments = [(v.target, best_values[v.target], v.unit) for v in variables]
    return OptimizeResult(
        values=best_values,
        objective=best_objective,
        baseline=baseline,
        evaluations=len(history),
        success=bool(res.success),
        message=str(res.message),
        history=history,
        setup=with_values(setup, assignments),
    )
