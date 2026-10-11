import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sequoia.data import Duration
from sequoia.errors import SequoiaError

DEFAULT_SEARCH = 50e-9
MAX_CHUNK_PAIRS = 5_000_000


class CorrelationError(SequoiaError):
    pass


def _duration(value: Any, what: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    try:
        from pydantic import TypeAdapter

        return TypeAdapter(Duration).validate_python(value).si
    except Exception as exc:
        raise CorrelationError(f"{what}: expected a duration such as '1 ns'") from exc


def delays_within(t_a: np.ndarray, t_b: np.ndarray, half_range: float):
    """Yield arrays of ``t_a - t_b`` for every pair closer than ``half_range`` (sorted inputs)."""
    lo = np.searchsorted(t_b, t_a - half_range, side="left")
    hi = np.searchsorted(t_b, t_a + half_range, side="right")
    counts = hi - lo
    cum = np.cumsum(counts)
    start = 0
    n = len(t_a)
    while start < n:
        base = cum[start - 1] if start else 0
        stop = int(np.searchsorted(cum, base + MAX_CHUNK_PAIRS, side="right"))
        stop = max(stop, start + 1)
        c = counts[start:stop]
        total = int(c.sum())
        if total:
            owner = np.repeat(np.arange(start, stop), c)
            offset = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
            yield t_a[owner] - t_b[lo[owner] + offset]
        start = stop


def delay_histogram(
    t_a: np.ndarray,
    t_b: np.ndarray,
    bin_width: float,
    half_range: float,
    center: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of ``t_a - t_b`` around ``center``; returns (bin centres, counts)."""
    nbins = max(int(round(2 * half_range / bin_width)), 1)
    edges = center - half_range + np.arange(nbins + 1) * (2 * half_range / nbins)
    counts = np.zeros(nbins, dtype=np.int64)
    for delays in delays_within(t_a, t_b, half_range + abs(center)):
        counts += np.histogram(delays, bins=edges)[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts


def count_in_window(t_a: np.ndarray, t_b: np.ndarray, window: float, center: float) -> int:
    lo = np.searchsorted(t_b, t_a - center - window / 2.0, side="left")
    hi = np.searchsorted(t_b, t_a - center + window / 2.0, side="right")
    return int(np.sum(hi - lo))


def per_event_counts(
    reference: np.ndarray, other: np.ndarray, window: float, center: float
) -> np.ndarray:
    """For each reference event, the number of ``other`` events with ``other - reference`` in the
    window around ``-center`` (so that channel order matches ``t_ref - t_other``)."""
    lo = np.searchsorted(other, reference - center - window / 2.0, side="left")
    hi = np.searchsorted(other, reference - center + window / 2.0, side="right")
    return hi - lo


def find_peak(
    t_a: np.ndarray, t_b: np.ndarray, window: float, search: float = DEFAULT_SEARCH
) -> float:
    """Delay of the correlation peak: the window position holding the most coincidences."""
    width = max(window / 4.0, search / 4000.0)
    centres, counts = delay_histogram(t_a, t_b, width, search)
    if counts.sum() == 0:
        return 0.0
    span = max(int(round(window / width)), 1)
    kernel = np.ones(span)
    smooth = np.convolve(counts, kernel, mode="same")
    k = int(np.argmax(smooth))
    lo, hi = max(k - span // 2, 0), min(k + span // 2 + 1, len(counts))
    weights = counts[lo:hi].astype(float)
    if weights.sum() == 0:
        return float(centres[k])
    return float(np.sum(centres[lo:hi] * weights) / weights.sum())


def _arr(tags: Mapping[str, np.ndarray], name: str, analysis: str) -> np.ndarray:
    if name not in tags:
        raise CorrelationError(
            f"analysis '{analysis}': unknown channel '{name}' (have {sorted(tags)})"
        )
    return tags[name]


def coincidences(
    t_a: np.ndarray,
    t_b: np.ndarray,
    duration: float,
    window: float,
    center: float | str = "auto",
    *,
    accidentals: str = "singles",
    sideband: float | None = None,
    search: float = DEFAULT_SEARCH,
) -> dict[str, Any]:
    """Coincidences in a window with accidentals from the singles product or sidebands."""
    if window <= 0:
        raise CorrelationError("the coincidence window must be positive")
    peak = find_peak(t_a, t_b, window, search) if center == "auto" else float(center)
    n_in = count_in_window(t_a, t_b, window, peak)
    s_a, s_b = len(t_a) / duration, len(t_b) / duration
    if accidentals == "sideband":
        offset = sideband if sideband is not None else max(10 * window, 20e-9)
        side = [count_in_window(t_a, t_b, window, peak + k * offset) for k in (-1, 1)]
        acc_counts = float(np.mean(side))
        var_acc = float(np.sum(side)) / 4.0
    else:
        acc_counts = s_a * s_b * window * duration
        var_acc = 0.0
    true_counts = n_in - acc_counts
    sigma = math.sqrt(max(n_in, 0) + var_acc) / duration
    acc = acc_counts / duration
    true = true_counts / duration
    return {
        "type": "coincidences",
        "window_s": window,
        "center_s": peak,
        "counts": n_in,
        "total_hz": n_in / duration,
        "accidental_hz": acc,
        "true_hz": true,
        "true_sigma_hz": sigma,
        "car": true / acc if acc > 0 else math.inf,
        "car_total": n_in / duration / acc if acc > 0 else math.inf,
        "singles_a_hz": s_a,
        "singles_b_hz": s_b,
        "heralding_a_given_b": true / s_b if s_b > 0 else 0.0,
        "heralding_b_given_a": true / s_a if s_a > 0 else 0.0,
    }


def heralded_g2(
    t_h: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
    duration: float,
    window: float,
    center_a: float | str = "auto",
    center_b: float | str = "auto",
    search: float = DEFAULT_SEARCH,
) -> dict[str, Any]:
    """Heralded second-order correlation g2(0) = N_hab N_h / (N_ha N_hb) from triples."""
    ca = find_peak(t_a, t_h, window, search) if center_a == "auto" else float(center_a)
    cb = find_peak(t_b, t_h, window, search) if center_b == "auto" else float(center_b)
    n_a = per_event_counts(t_h, t_a, window, -ca)
    n_b = per_event_counts(t_h, t_b, window, -cb)
    n_ha, n_hb = int(n_a.sum()), int(n_b.sum())
    n_hab = int(np.sum(n_a.astype(np.int64) * n_b))
    n_h = len(t_h)
    g2 = n_hab * n_h / (n_ha * n_hb) if n_ha > 0 and n_hb > 0 else math.nan
    sigma = g2 / math.sqrt(n_hab) if n_hab > 0 else math.nan
    return {
        "type": "heralded_g2",
        "window_s": window,
        "herald_singles_hz": n_h / duration,
        "rate_ha_hz": n_ha / duration,
        "rate_hb_hz": n_hb / duration,
        "triples": n_hab,
        "triples_hz": n_hab / duration,
        "triples_sigma_hz": math.sqrt(max(n_hab, 1)) / duration,
        "g2_heralded": g2,
        "g2_sigma": sigma,
        "center_a_s": ca,
        "center_b_s": cb,
    }


def g2_histogram(
    t_a: np.ndarray,
    t_b: np.ndarray,
    duration: float,
    half_range: float,
    bins: int,
    center: float = 0.0,
) -> dict[str, Any]:
    width = 2 * half_range / bins
    centres, counts = delay_histogram(t_a, t_b, width, half_range, center)
    s_a, s_b = len(t_a) / duration, len(t_b) / duration
    expected = s_a * s_b * width * duration
    g2 = counts / expected if expected > 0 else np.full(len(counts), math.nan)
    return {
        "type": "g2",
        "tau_s": centres,
        "counts": counts,
        "rate_hz": counts / duration,
        "g2": g2,
        "bin_width_s": width,
        "center_s": center,
        "g2_peak": float(np.nanmax(g2)) if len(g2) else math.nan,
    }


def analyse_tags(
    channels: Mapping[str, np.ndarray],
    duration: float,
    specs: Sequence[Mapping[str, Any]],
    repetition_rates: Sequence[float] = (),
) -> dict[str, dict[str, Any]]:
    """Apply the setup's ``analyses`` entries to a time-tag record.

    ``coincidences``, ``heralded_g2``, ``g2`` and ``histogram`` entries use the same keys as the
    rate engine. Accidentals come from sidebands one pulse period away for pulsed sources and
    from the singles product for CW sources.
    """
    out: dict[str, dict[str, Any]] = {}
    rep = max(repetition_rates) if repetition_rates else None
    for spec in specs:
        name = str(spec.get("name", spec.get("type", "analysis")))
        kind = spec.get("type")
        try:
            if kind == "coincidences":
                a, b = spec["channels"]
                window = _duration(spec["window"], f"analysis '{name}': window")
                center = spec.get("center", "auto")
                center = (
                    "auto"
                    if isinstance(center, str) and center.lower() == "auto"
                    else _duration(center, f"analysis '{name}': center")
                )
                mode = "sideband" if rep else "singles"
                result = coincidences(
                    _arr(channels, a, name),
                    _arr(channels, b, name),
                    duration,
                    window,
                    center,
                    accidentals=mode,
                    sideband=(1.0 / rep) if rep else None,
                )
                result["channels"] = [a, b]
            elif kind == "heralded_g2":
                window = _duration(spec["window"], f"analysis '{name}': window")
                result = heralded_g2(
                    _arr(channels, spec["herald"], name),
                    _arr(channels, spec["a"], name),
                    _arr(channels, spec["b"], name),
                    duration,
                    window,
                )
                result.update(herald=spec["herald"], a=spec["a"], b=spec["b"])
            elif kind in ("g2", "histogram"):
                a, b = spec["channels"]
                half = _duration(spec.get("range", "10 ns"), f"analysis '{name}': range")
                result = g2_histogram(
                    _arr(channels, a, name),
                    _arr(channels, b, name),
                    duration,
                    half,
                    int(spec.get("bins", 100)),
                    _duration(spec.get("center", 0.0), f"analysis '{name}': center"),
                )
                result["channels"] = [a, b]
            else:
                raise CorrelationError(f"analysis '{name}': unknown type {kind!r}")
        except KeyError as exc:
            raise CorrelationError(f"analysis '{name}': missing key {exc}") from exc
        out[name] = result
    return out


def read_tags(path: Path | str) -> tuple[dict[str, np.ndarray], float | None]:
    """Read time tags from ``.npz`` (one array of seconds per channel) or a CSV of
    ``channel,time_s`` rows. Returns the sorted arrays and the duration if the file has one."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            duration = float(data["duration"]) if "duration" in data.files else None
            channels = {
                k: np.sort(np.asarray(data[k], dtype=float)) for k in data.files if k != "duration"
            }
        return channels, duration
    columns: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                value = float(row[1])
            except ValueError:
                continue
            columns.setdefault(row[0].strip(), []).append(value)
    if not columns:
        raise CorrelationError(f"{path}: no 'channel,time_s' rows found")
    return {k: np.sort(np.asarray(v)) for k, v in columns.items()}, None


def write_tags(path: Path | str, channels: Mapping[str, np.ndarray], duration: float) -> None:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise CorrelationError("time tags are written as .npz")
    np.savez_compressed(path, duration=duration, **{k: np.asarray(v) for k, v in channels.items()})
