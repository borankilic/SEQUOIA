import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sequoia.analysis.metrics import chsh_s, correlation_from_rates
from sequoia.analysis.sweep import SweepError, check_axis_unit, evaluate, with_values
from sequoia.engines.cache import RunCache
from sequoia.library.store import Library
from sequoia.setup.model import Setup

DEFAULT_OUTCOMES = (("tt", "tr"), ("rt", "rr"))


@dataclass
class ChshResult:
    rates: np.ndarray
    correlations: np.ndarray
    s: float
    s_sigma: float
    quantity: str
    integration_time: float
    settings_a: tuple[float, float]
    settings_b: tuple[float, float]
    unit: str
    notes: list[str] = field(default_factory=list)

    @property
    def violates(self) -> bool:
        return self.s - 2.0 > 3.0 * self.s_sigma if self.s_sigma > 0 else self.s > 2.0


def run_chsh(
    setup: Setup,
    library: Library,
    target_a: str,
    target_b: str,
    settings_a: Sequence[float],
    settings_b: Sequence[float],
    unit: str = "deg",
    outcomes: Sequence[Sequence[str]] = DEFAULT_OUTCOMES,
    quantity: str = "true_hz",
    integration_time: float = 10.0,
    cache: RunCache | None = None,
    should_cancel=None,
) -> ChshResult:
    """CHSH S from four analyser-setting pairs, using the rate engine's coincidence analyses.

    ``outcomes[i][j]`` names the coincidence analysis between analyser output i of arm a and
    output j of arm b (output 0 transmitted, 1 the orthogonal port). ``settings_*`` are the two
    values of the analyser parameters ``target_*`` (for a half-wave plate, half the polarizer
    angle). ``quantity`` is ``true_hz`` (accidentals subtracted) or ``total_hz``.
    """
    if len(settings_a) != 2 or len(settings_b) != 2:
        raise SweepError("a CHSH test needs two settings per arm")
    if quantity not in ("true_hz", "total_hz"):
        raise SweepError("quantity must be 'true_hz' or 'total_hz'")
    for target, value in ((target_a, settings_a[0]), (target_b, settings_b[0])):
        check_axis_unit(setup, target, unit)
        with_values(setup, [(target, float(value), unit)])
    cache = cache if cache is not None else RunCache()
    rates = np.zeros((2, 2, 2, 2))
    notes: list[str] = []
    for i, a in enumerate(settings_a):
        for j, b in enumerate(settings_b):
            point = with_values(setup, [(target_a, float(a), unit), (target_b, float(b), unit)])
            outcome = evaluate(point, library, cache=cache, should_cancel=should_cancel)
            if outcome.rates is None or outcome.has_errors:
                first = next((x for x in outcome.issues if x.severity == "error"), None)
                raise SweepError(first.message if first else "simulation failed")
            for k in range(2):
                for m in range(2):
                    name = outcomes[k][m]
                    analysis = outcome.rates.analyses.get(name)
                    if analysis is None or quantity not in analysis:
                        raise SweepError(
                            f"analysis '{name}' with a '{quantity}' value is not in the setup; "
                            "add coincidence analyses for all four detector pairs"
                        )
                    rates[i, j, k, m] = analysis[quantity]
    e = np.zeros((2, 2))
    variance = 0.0
    for i in range(2):
        for j in range(2):
            r = rates[i, j]
            e[i, j] = correlation_from_rates(r[0, 0], r[0, 1], r[1, 0], r[1, 1])
            counts = float(r.sum()) * integration_time
            if counts <= 0:
                notes.append(f"no coincidences for setting pair ({i}, {j})")
                variance = math.inf
            else:
                variance += (1.0 - e[i, j] ** 2) / counts
    s = chsh_s(e[0, 0], e[0, 1], e[1, 0], e[1, 1])
    return ChshResult(
        rates,
        e,
        float(s),
        math.sqrt(variance) if math.isfinite(variance) else math.nan,
        quantity,
        integration_time,
        (float(settings_a[0]), float(settings_a[1])),
        (float(settings_b[0]), float(settings_b[1])),
        unit,
        notes,
    )


def format_chsh(result: ChshResult) -> str:
    lines = [
        f"CHSH  S = {result.s:.4f} +- {result.s_sigma:.4f}  "
        f"({'violates' if result.violates else 'does not violate'} S <= 2; "
        f"Tsirelson bound 2.8284; {result.quantity}, {result.integration_time:g} s)",
        f"settings a = {result.settings_a} {result.unit}, b = {result.settings_b} {result.unit}",
        "E(a_i, b_j):",
    ]
    for i in range(2):
        lines.append("  " + "  ".join(f"{result.correlations[i, j]:+.4f}" for j in range(2)))
    lines += [f"note: {n}" for n in result.notes]
    return "\n".join(lines)
