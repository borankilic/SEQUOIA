import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sequoia.analysis.metrics import ratio
from sequoia.data import Duration
from sequoia.errors import SequoiaError

DEFAULT_SEARCH = 50e-9
WIDE_SEARCHES = (1e-6, 20e-6)
TAG_FORMAT_VERSION = 2
RESERVED_KEYS = {"duration", "repetition_rates", "format_version", "meta_json"}
MAX_CHUNK_PAIRS = 5_000_000
CW_SIDEBAND_MIN = 20e-9
CW_SIDEBAND_RECORD = 100
CW_OFF_PEAK = "sideband (CW off-peak)"


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


def gate(
    t: np.ndarray, herald: Sequence[np.ndarray], veto: Sequence[np.ndarray], window: float
) -> np.ndarray:
    """Mask of the events of ``t`` with a tag of every herald channel and of no veto channel
    within +-window/2."""
    keep = np.ones(len(t), dtype=bool)
    for h in herald:
        keep &= per_event_counts(t, h, window, 0.0) > 0
    for v in veto:
        keep &= per_event_counts(t, v, window, 0.0) == 0
    return keep


def cw_sideband_offset(window: float, width: float | None = None) -> float:
    return max(10.0 * window, CW_SIDEBAND_MIN, 5.0 * width if width else 0.0)


def locate_peak(
    t_a: np.ndarray,
    t_b: np.ndarray,
    window: float,
    search: float = DEFAULT_SEARCH,
    center: float = 0.0,
) -> tuple[float, bool]:
    """Delay of the correlation peak and whether it stands out of the background."""
    width = max(window / 4.0, search / 4000.0)
    centres, counts = delay_histogram(t_a, t_b, width, search, center)
    if counts.sum() == 0:
        return 0.0, False
    span = max(int(round(window / width)), 1)
    kernel = np.ones(span)
    smooth = np.convolve(counts, kernel, mode="same")
    k = int(np.argmax(smooth))
    floor = float(np.median(smooth))
    significant = bool(smooth[k] >= 3 and smooth[k] > floor + 5.0 * math.sqrt(floor + 1.0))
    lo, hi = max(k - span // 2, 0), min(k + span // 2 + 1, len(counts))
    weights = counts[lo:hi].astype(float)
    if weights.sum() == 0:
        return float(centres[k]), significant
    return float(np.sum(centres[lo:hi] * weights) / weights.sum()), significant


def find_peak(
    t_a: np.ndarray, t_b: np.ndarray, window: float, search: float = DEFAULT_SEARCH
) -> float:
    """Delay of the correlation peak: the window position holding the most coincidences."""
    return locate_peak(t_a, t_b, window, search)[0]


def find_peak_wide(
    t_a: np.ndarray,
    t_b: np.ndarray,
    window: float,
    search: float = DEFAULT_SEARCH,
    widen: bool = True,
) -> tuple[float, bool, float]:
    """Peak search that widens from ``search`` through ``WIDE_SEARCHES`` until a peak stands out."""
    ranges = [search] + ([w for w in WIDE_SEARCHES if w > search] if widen else [])
    best = (0.0, False, search)
    for half in ranges:
        centre, found = locate_peak(t_a, t_b, window, half)
        if found:
            coarse = max(window / 4.0, half / 4000.0)
            if coarse > window / 4.0:
                centre, _ = locate_peak(t_a, t_b, window, 2.0 * coarse, centre)
            return centre, True, half
        best = (centre, False, half) if half == ranges[0] else best
    return best


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
    widen: bool = True,
) -> dict[str, Any]:
    """Coincidences in a window with accidentals from the singles product or sidebands."""
    if window <= 0:
        raise CorrelationError("the coincidence window must be positive")
    if center == "auto":
        peak, found, used = find_peak_wide(t_a, t_b, window, search, widen=widen)
    else:
        peak, found, used = float(center), None, None
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
        "car": ratio(true, acc),
        "car_total": ratio(n_in / duration, acc),
        "peak_found": found,
        "center_mode": "search" if center == "auto" else "fixed",
        "search_s": used,
        "accidental_model": accidentals,
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
    widen: bool = True,
) -> dict[str, Any]:
    """Heralded second-order correlation g2(0) = N_hab N_h / (N_ha N_hb) from triples.

    Centers are the delays ``t_a - t_h`` and ``t_b - t_h`` of the herald correlation peaks.
    """
    if center_a == "auto":
        ca, found_a, used_a = find_peak_wide(t_a, t_h, window, search, widen=widen)
    else:
        ca, found_a, used_a = float(center_a), None, None
    if center_b == "auto":
        cb, found_b, used_b = find_peak_wide(t_b, t_h, window, search, widen=widen)
    else:
        cb, found_b, used_b = float(center_b), None, None
    modes = {"search" if c == "auto" else "fixed" for c in (center_a, center_b)}
    searched = [u for u in (used_a, used_b) if u is not None]
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
        "peak_found_a": found_a,
        "peak_found_b": found_b,
        "center_mode": modes.pop() if len(modes) == 1 else "mixed",
        "search_s": max(searched) if searched else None,
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


TIMING_NOTES = {
    "unknown": "no single pulse clock is known for {names} (mixed or unknown timing); the CW "
    "singles estimator was used and is not valid for pulsed light",
    "mixed": "{names} mix pulsed and CW light; the CW singles estimator was used and is not "
    "valid for the pulsed part",
    "unequal": "{names} have unequal pulse clocks; the CW singles estimator was used and is not "
    "valid for pulsed light",
    "unsynchronized": "{names} are clocked by different sources at the same rate, which need "
    "not be synchronized; the CW singles estimator was used and is not valid for pulsed light",
    "legacy": "the record has no per-channel clock; the fastest source clock was applied to "
    "{names}",
}


def _clock(
    rates: Mapping[str, float] | Sequence[float],
    names: Sequence[str],
    ids: Mapping[str, str] | None = None,
) -> tuple[float | None, str]:
    """Common pulse clock of ``names`` and the timing class: ``pulsed``, ``cw``, ``legacy``
    (a plain list of source rates) or a reason why no clock applies."""
    if not isinstance(rates, Mapping):
        return (float(max(rates)), "legacy") if len(rates) else (None, "cw")
    values = [rates.get(n) for n in names]
    if any(v is None for v in values):
        return None, "unknown"
    values = [float(v) for v in values]
    if all(v == 0.0 for v in values):
        return None, "cw"
    if any(v == 0.0 for v in values):
        return None, "mixed"
    if max(values) > min(values) * (1.0 + 1e-9):
        return None, "unequal"
    if ids is not None:
        known = {ids.get(n) for n in names}
        if None in known or len(known) > 1:
            return None, "unsynchronized"
    return values[0], "pulsed"


def _timing(result: dict[str, Any], timing: str, names: Sequence[str]) -> None:
    result["timing"] = timing
    if timing in TIMING_NOTES:
        result["timing_note"] = TIMING_NOTES[timing].format(names=" and ".join(names))


def _center(spec: Mapping[str, Any], key: str, name: str) -> float | str:
    value = spec.get(key, "auto")
    mode = spec.get("mode")
    if mode not in (None, "search", "fixed"):
        raise CorrelationError(f"analysis '{name}': mode must be 'search' or 'fixed', got {mode!r}")
    if isinstance(value, str) and value.lower() == "auto":
        if mode == "fixed":
            raise CorrelationError(
                f"analysis '{name}': mode 'fixed' needs an explicit {key} (see confirm_centers)"
            )
        return "auto"
    if mode == "search":
        raise CorrelationError(f"analysis '{name}': mode 'search' cannot use a fixed {key}")
    return _duration(value, f"analysis '{name}': {key}")


def confirm_centers(
    search_results: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Analysis specs with the centres found by a search frozen (``mode: fixed``).

    Search on one data set, then measure another with these specs so that the centre is not
    re-selected from the data being measured (confirmatory analysis). Without ``specs`` only the
    coincidence and heralded-g2 specs are rebuilt from the results; other given specs pass
    unchanged.
    """
    if specs is None:
        specs = []
        for name, r in search_results.items():
            spec: dict[str, Any] = {"name": name, "type": r["type"], "window": r.get("window_s")}
            if r["type"] == "coincidences":
                spec["channels"] = list(r["channels"])
            elif r["type"] == "heralded_g2":
                spec.update(herald=r["herald"], a=r["a"], b=r["b"])
            else:
                continue
            specs.append(spec)
    out = []
    for spec in specs:
        spec = dict(spec)
        name = str(spec.get("name", spec.get("type", "analysis")))
        kind = spec.get("type")
        if kind in ("coincidences", "heralded_g2"):
            r = search_results.get(name)
            if r is None:
                raise CorrelationError(f"analysis '{name}': no search result to take a centre from")
            keys = (
                [("center", "center_s", "peak_found")]
                if kind == "coincidences"
                else [
                    ("center_a", "center_a_s", "peak_found_a"),
                    ("center_b", "center_b_s", "peak_found_b"),
                ]
            )
            for key, value, found in keys:
                if r.get(found) is False:
                    raise CorrelationError(
                        f"analysis '{name}': the search found no peak for {key}; a frozen centre "
                        "would be arbitrary"
                    )
                spec[key] = float(r[value])
            spec["mode"] = "fixed"
            spec["confirmed_from"] = {
                "search_s": r.get("search_s"),
                "center_mode": r.get("center_mode"),
            }
            spec.pop("search", None)
        out.append(spec)
    return out


def analyse_tags(
    channels: Mapping[str, np.ndarray],
    duration: float,
    specs: Sequence[Mapping[str, Any]],
    repetition_rates: Mapping[str, float] | Sequence[float] = (),
    clock_ids: Mapping[str, str] | None = None,
    accidentals: str = "auto",
    conditions: Mapping[str, tuple[Sequence[str], Sequence[str]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply the setup's ``analyses`` entries to a time-tag record.

    ``coincidences``, ``heralded_g2``, ``g2`` and ``histogram`` entries use the same keys as the
    rate engine. Accidentals come from sidebands one pulse period away for pulsed sources and
    from the singles product for CW sources. ``repetition_rates`` is either a sequence (the
    fastest clock applies to every analysis, legacy records) or a ``{channel: rate_hz}`` map, in
    which 0 marks CW light and a missing channel has no single known clock. ``clock_ids`` names
    the clock of each pulsed channel; equal rates from different clocks are not treated as
    synchronized. Without a common clock the CW singles estimator is used and the result carries
    ``timing`` and ``timing_note`` saying why.

    Optional keys: ``center`` (coincidences), ``center_a`` / ``center_b`` (heralded g2),
    ``search`` (half range of the automatic peak search) and ``mode``: ``search`` (centres
    found in these data) or ``fixed`` (explicit centres, e.g. from ``confirm_centers``, for a
    confirmatory measurement). Without ``mode`` a numeric centre is fixed and ``auto`` searches.
    Results record ``center_mode`` and ``search_s`` (None when nothing was searched).

    ``accidentals="sideband"`` on CW light (every channel clock 0) uses off-peak windows at
    +-``cw_sideband_offset`` (10 x window, 20 ns, 5 x the spec's ``correlation_width``), unless
    the record is shorter than ``CW_SIDEBAND_RECORD`` offsets. A spec's ``condition`` names an
    entry of ``conditions`` (herald channels, veto channels): only events with a tag on every
    herald channel and none on a veto channel within +-window/2 are counted.
    """
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name = str(spec.get("name", spec.get("type", "analysis")))
        kind = spec.get("type")
        try:
            search = (
                _duration(spec["search"], f"analysis '{name}': search")
                if "search" in spec
                else DEFAULT_SEARCH
            )
            widen = "search" not in spec
            used = (
                [spec["herald"], spec["a"], spec["b"]]
                if kind == "heralded_g2"
                else list(spec.get("channels") or [])
            )
            tags, gated = _gated(channels, used, spec, conditions, name)
            if kind == "coincidences":
                a, b = spec["channels"]
                window = _duration(spec["window"], f"analysis '{name}': window")
                rep, timing = _clock(repetition_rates, [a, b], clock_ids)
                offset, short = None, False
                if not rep and timing == "cw" and accidentals == "sideband":
                    width = spec.get("correlation_width")
                    width = (
                        _duration(width, f"analysis '{name}': correlation_width")
                        if width is not None
                        else None
                    )
                    offset = cw_sideband_offset(window, width)
                    short = duration < CW_SIDEBAND_RECORD * offset
                cw_side = offset is not None and not short
                result = coincidences(
                    _arr(tags, a, name),
                    _arr(tags, b, name),
                    duration,
                    window,
                    _center(spec, "center", name),
                    accidentals="sideband" if rep or cw_side else "singles",
                    sideband=(1.0 / rep) if rep else offset,
                    search=search,
                    widen=widen,
                )
                result["channels"] = [a, b]
                _timing(result, timing, [a, b])
                if cw_side:
                    result["accidental_model"] = CW_OFF_PEAK
                    result["sideband_offsets_s"] = [-offset, offset]
                    result["timing_note"] = (
                        f"CW: accidentals from off-peak windows at +-{offset * 1e9:.4g} ns "
                        "(mean of two sidebands)"
                    )
                elif short:
                    result["timing_note"] = (
                        f"CW: the record ({duration:.4g} s) is shorter than "
                        f"{CW_SIDEBAND_RECORD} x the off-peak offset ({offset * 1e9:.4g} ns); "
                        "singles product used"
                    )
            elif kind == "heralded_g2":
                window = _duration(spec["window"], f"analysis '{name}': window")
                result = heralded_g2(
                    _arr(tags, spec["herald"], name),
                    _arr(tags, spec["a"], name),
                    _arr(tags, spec["b"], name),
                    duration,
                    window,
                    _center(spec, "center_a", name),
                    _center(spec, "center_b", name),
                    search,
                    widen,
                )
                result.update(herald=spec["herald"], a=spec["a"], b=spec["b"])
            elif kind in ("g2", "histogram"):
                a, b = spec["channels"]
                half = _duration(spec.get("range", "10 ns"), f"analysis '{name}': range")
                result = g2_histogram(
                    _arr(tags, a, name),
                    _arr(tags, b, name),
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
        if "confirmed_from" in spec:
            result["confirmed_from"] = dict(spec["confirmed_from"])
        if gated is not None:
            result["condition"] = gated
        out[name] = result
    return out


def _gated(
    channels: Mapping[str, np.ndarray],
    used: Sequence[str],
    spec: Mapping[str, Any],
    conditions: Mapping[str, tuple[Sequence[str], Sequence[str]]] | None,
    name: str,
) -> tuple[Mapping[str, np.ndarray], dict[str, Any] | None]:
    cond = spec.get("condition")
    if cond is None:
        return channels, None
    if cond not in (conditions or {}):
        raise CorrelationError(
            f"analysis '{name}': unknown condition '{cond}' (have {sorted(conditions or {})})"
        )
    if spec.get("window") is None:
        raise CorrelationError(f"analysis '{name}': condition '{cond}' needs a window")
    window = _duration(spec["window"], f"analysis '{name}': window")
    herald, veto = conditions[cond]
    for ch in [*herald, *veto]:
        if ch not in channels:
            raise CorrelationError(
                f"analysis '{name}': condition '{cond}': unknown channel '{ch}' "
                f"(have {sorted(channels)})"
            )
    out = dict(channels)
    kept: dict[str, int] = {}
    events: dict[str, int] = {}
    for ch in dict.fromkeys(used):
        t = _arr(channels, ch, name)
        mask = gate(
            t,
            [channels[h] for h in herald if h != ch],
            [channels[v] for v in veto if v != ch],
            window,
        )
        out[ch] = t[mask]
        kept[ch], events[ch] = int(mask.sum()), len(t)
    return out, {
        "name": cond,
        "herald": list(herald),
        "veto": list(veto),
        "window_s": window,
        "kept": kept,
        "events": events,
    }


@dataclass
class TagRecord:
    channels: dict[str, np.ndarray]
    duration: float | None
    repetition_rates: dict[str, float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    format_version: int = 1
    clock_ids: dict[str, str] | None = None

    @property
    def has_clock(self) -> bool:
        return self.repetition_rates is not None


def read_tag_record(path: Path | str) -> TagRecord:
    """Read time tags from ``.npz`` (version 1 or 2) or a CSV of ``channel,time_s`` rows."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            duration = float(data["duration"]) if "duration" in files else None
            version = int(data["format_version"]) if "format_version" in files else 1
            meta: dict[str, Any] = {}
            rates: dict[str, float] | None = None
            ids: dict[str, str] | None = None
            if "meta_json" in files:
                meta = json.loads(str(data["meta_json"]))
                if "clock" in meta:
                    rates = {str(k): float(v) for k, v in meta.pop("clock").items()}
                if "clock_ids" in meta:
                    ids = {str(k): str(v) for k, v in meta.pop("clock_ids").items()}
            channels = {
                k: np.sort(np.asarray(data[k], dtype=float))
                for k in data.files
                if k not in RESERVED_KEYS
            }
        return TagRecord(channels, duration, rates, meta, version, ids)
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
    return TagRecord({k: np.sort(np.asarray(v)) for k, v in columns.items()}, None)


def read_tags(path: Path | str) -> tuple[dict[str, np.ndarray], float | None]:
    record = read_tag_record(path)
    return record.channels, record.duration


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_tags(
    path: Path | str,
    channels: Mapping[str, np.ndarray],
    duration: float,
    repetition_rates: Mapping[str, float] | Sequence[float] = (),
    meta: Mapping[str, Any] | None = None,
    clock_ids: Mapping[str, str] | None = None,
    analyses: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Write a version-2 tag file; the pulse clock and run metadata travel with the arrays.
    A clock map is written as given, an empty one included (no channel has a valid clock).
    ``analyses`` (results of ``analyse_tags``) stores their centres, centre mode, search ranges
    and windows in ``meta["analyses"]``."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise CorrelationError("time tags are written as .npz")
    clash = RESERVED_KEYS & set(channels)
    if clash:
        raise CorrelationError(f"channel names {sorted(clash)} are reserved in tag files")
    info = dict(meta or {})
    if isinstance(repetition_rates, Mapping):
        info["clock"] = {str(k): float(v) for k, v in repetition_rates.items()}
    elif len(repetition_rates):
        info["clock"] = {name: float(max(repetition_rates)) for name in channels}
    if clock_ids is not None:
        info["clock_ids"] = {str(k): str(v) for k, v in clock_ids.items()}
    if analyses is not None:
        keep = (
            "type",
            "window_s",
            "center_mode",
            "center_s",
            "center_a_s",
            "center_b_s",
            "search_s",
            "peak_found",
            "peak_found_a",
            "peak_found_b",
        )
        info["analyses"] = {
            str(n): {k: _plain(a[k]) for k in keep if k in a} for n, a in analyses.items()
        }
    payload = {k: np.asarray(v) for k, v in channels.items()}
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=path.parent or ".")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp,
            duration=duration,
            format_version=TAG_FORMAT_VERSION,
            meta_json=json.dumps(info, default=str),
            **payload,
        )
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
