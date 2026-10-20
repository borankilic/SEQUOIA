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
    """A model forecast of CHSH S, not a measured rejection of local models.

    ``rates[i, j, k, m]`` are the coincidence rates used for S (``quantity``); ``s_raw`` /
    ``s_corrected`` are S from total and accidental-subtracted rates when both are available.
    ``s_sigma`` is the anticipated counting spread of the estimator for ``integration_time``
    seconds per setting pair (four pairs, so 4x that in total), or None when no coincidences
    are predicted. ``violates`` compares |S| (two-sided, either output labelling) with the
    local bound 2 at three sigma and is None when the spread is undefined."""

    rates: np.ndarray
    correlations: np.ndarray
    s: float
    s_sigma: float | None
    quantity: str
    integration_time: float
    settings_a: tuple[float, float]
    settings_b: tuple[float, float]
    unit: str
    notes: list[str] = field(default_factory=list)
    s_raw: float | None = None
    s_corrected: float | None = None
    correlations_corrected: np.ndarray | None = None
    sidebands: int = 2
    total_rates: np.ndarray | None = None
    accidental_rates: np.ndarray | None = None

    @property
    def significance(self) -> float | None:
        if self.s_sigma is None or not self.s_sigma > 0 or not math.isfinite(self.s_sigma):
            return None
        return (abs(self.s) - 2.0) / self.s_sigma

    @property
    def violates(self) -> bool | None:
        z = self.significance
        return None if z is None else z > 3.0


def _finite(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise SweepError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(v):
        raise SweepError(f"{name} must be finite, got {v}")
    return v


def _correlation_variance(total: np.ndarray, acc: np.ndarray, sidebands: int, e: float) -> float:
    # E = sum_k s_k C_k / sum_k C_k with C_k = R_k - A_k; R_k Poisson, the accidental estimate
    # from ``sidebands`` sideband windows has variance A_k / sidebands (0: taken as exact)
    signs = np.array([1.0, -1.0, -1.0, 1.0])
    c = total - acc
    norm = float(c.sum())
    var_c = total + (acc / sidebands if sidebands > 0 else 0.0)
    return float(np.sum((signs - e) ** 2 * var_c)) / norm**2


def run_chsh(
    setup: Setup,
    library: Library,
    target_a: str,
    target_b: str,
    settings_a: Sequence[float],
    settings_b: Sequence[float],
    unit: str = "deg",
    outcomes: Sequence[Sequence[str]] = DEFAULT_OUTCOMES,
    quantity: str = "total_hz",
    integration_time: float = 10.0,
    cache: RunCache | None = None,
    should_cancel=None,
    sidebands: int = 2,
) -> ChshResult:
    """CHSH S from four analyser-setting pairs, using the rate engine's coincidence analyses.

    ``outcomes[i][j]`` names the coincidence analysis between analyser output i of arm a and
    output j of arm b (output 0 transmitted, 1 the orthogonal port). ``settings_*`` are the two
    values of the analyser parameters ``target_*`` (for a half-wave plate, half the polarizer
    angle). ``quantity`` is ``total_hz`` (raw coincidences, default) or ``true_hz``
    (accidentals subtracted, estimated from ``sidebands`` sideband windows). Coincidences are
    post-selected on both photons being detected (fair-sampling assumption).
    ``integration_time`` is per setting pair.
    """
    if len(settings_a) != 2 or len(settings_b) != 2:
        raise SweepError("a CHSH test needs two settings per arm")
    if quantity not in ("true_hz", "total_hz"):
        raise SweepError("quantity must be 'true_hz' or 'total_hz'")
    integration_time = _finite("integration_time", integration_time)
    if integration_time <= 0.0:
        raise SweepError(f"integration_time must be positive, got {integration_time}")
    settings_a = [_finite("arm-a setting", v) for v in settings_a]
    settings_b = [_finite("arm-b setting", v) for v in settings_b]
    if int(sidebands) != sidebands or sidebands < 0:
        raise SweepError("sidebands must be a non-negative integer")
    for target, value in ((target_a, settings_a[0]), (target_b, settings_b[0])):
        check_axis_unit(setup, target, unit)
        with_values(setup, [(target, float(value), unit)])
    cache = cache if cache is not None else RunCache()
    total = np.zeros((2, 2, 2, 2))
    acc = np.zeros((2, 2, 2, 2))
    has_acc = True
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
                    if "total_hz" in analysis and "accidental_hz" in analysis:
                        total[i, j, k, m] = analysis["total_hz"]
                        acc[i, j, k, m] = analysis["accidental_hz"]
                    else:
                        has_acc = False
                        total[i, j, k, m] = analysis[quantity]
    if not has_acc and quantity == "true_hz":
        raise SweepError("accidental-subtracted CHSH needs total_hz and accidental_hz analyses")
    corrected = total - acc
    rates = corrected if quantity == "true_hz" else total
    notes: list[str] = []

    def correlations(r: np.ndarray) -> np.ndarray | None:
        e = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                norm = float(r[i, j].sum())
                if not norm > 0:
                    return None
                q = r[i, j]
                e[i, j] = correlation_from_rates(q[0, 0], q[0, 1], q[1, 0], q[1, 1])
        return e

    e = correlations(rates)
    e_raw = correlations(total)
    e_cor = correlations(corrected) if has_acc else None
    variance: float | None = 0.0
    if e is None:
        notes.append("no coincidences predicted for at least one setting pair; S is undefined")
        e = np.full((2, 2), math.nan)
        variance = None
    else:
        for i in range(2):
            for j in range(2):
                counts = float(rates[i, j].sum()) * integration_time
                if not counts > 0:
                    variance = None
                    break
                a_part = acc[i, j].ravel() if quantity == "true_hz" else np.zeros(4)
                variance += _correlation_variance(
                    total[i, j].ravel() * integration_time,
                    a_part * integration_time,
                    int(sidebands),
                    float(e[i, j]),
                )
            if variance is None:
                break
    s = chsh_s(e[0, 0], e[0, 1], e[1, 0], e[1, 1])
    if quantity == "true_hz" and float(acc.sum()) > 0:
        notes.append(
            "accidentals subtracted; the subtraction's variance is included "
            f"({int(sidebands)} sideband windows)"
            if sidebands
            else "accidentals subtracted; the accidental estimate is taken as exact"
        )
    notes.append(
        "forecast from the model: coincidences are post-selected on two detections "
        "(fair-sampling assumption); this is not a measured rejection of local models"
    )
    return ChshResult(
        rates,
        e,
        float(s),
        math.sqrt(variance) if variance is not None else None,
        quantity,
        integration_time,
        (float(settings_a[0]), float(settings_a[1])),
        (float(settings_b[0]), float(settings_b[1])),
        unit,
        notes,
        None if e_raw is None else float(chsh_s(*e_raw.ravel())),
        None if e_cor is None else float(chsh_s(*e_cor.ravel())),
        e_cor,
        int(sidebands),
        total,
        acc if has_acc else None,
    )


def format_chsh(result: ChshResult) -> str:
    z = result.significance
    sigma = "undefined" if result.s_sigma is None else f"{result.s_sigma:.4f}"
    if z is None:
        verdict = "significance undefined (no counting uncertainty)"
    else:
        verdict = (
            f"|S| - 2 = {z:.1f} sigma: "
            f"{'would violate' if result.violates else 'would not violate'} |S| <= 2"
        )
    kind = "raw coincidences" if result.quantity == "total_hz" else "accidentals subtracted"
    lines = [
        f"CHSH forecast  S = {result.s:.4f}, anticipated counting sigma {sigma}  ({kind}; "
        f"{result.integration_time:g} s per setting pair, 4 pairs)",
        f"  {verdict}; Tsirelson bound 2.8284",
    ]
    if result.s_raw is not None and result.s_corrected is not None:
        lines.append(
            f"  S raw {result.s_raw:.4f}, S accidental-subtracted {result.s_corrected:.4f}"
        )
    lines += [
        f"settings a = {result.settings_a} {result.unit}, b = {result.settings_b} {result.unit}",
        "E(a_i, b_j):",
    ]
    for i in range(2):
        lines.append("  " + "  ".join(f"{result.correlations[i, j]:+.4f}" for j in range(2)))
    lines += [f"note: {n}" for n in result.notes]
    return "\n".join(lines)
