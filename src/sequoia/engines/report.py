import math

from sequoia.engines.classical import SimulationResult
from sequoia.engines.rates import RateResult


def _mm(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value * 1e3:.4g}"


def format_report(result: SimulationResult) -> str:
    lines = []
    header = (
        f"{'link':<10}{'kind':<11}{'from -> to':<28}{'P [mW]':>10}{'w_x [mm]':>10}"
        f"{'w_y [mm]':>10}{'R_x [m]':>10}{'DOP':>7}"
    )
    lines += ["Links", header, "-" * len(header)]
    for state in result.links.values():
        route = f"{state.src} -> {state.dst}"
        light = state.end
        if light is None:
            lines.append(f"{state.id:<10}{state.kind:<11}{route:<28}{'(no light)':>10}")
            continue
        m = light.mode
        radius = m.radius_x
        lines.append(
            f"{state.id:<10}{state.kind:<11}{route:<28}{light.power * 1e3:>10.4g}"
            f"{_mm(m.w_x):>10}{_mm(m.w_y):>10}"
            f"{('inf' if math.isinf(radius) else f'{radius:.4g}'):>10}{light.dop:>7.3f}"
        )
    fibers = [s for s in result.links.values() if "coupling_efficiency" in s.info]
    if fibers:
        lines += ["", "Fiber links"]
        for s in fibers:
            light = (
                f"classical light at {s.end.mean_wavelength * 1e9:.4g} nm"
                if s.end
                else ("classical light")
            )
            lines.append(
                f"  {s.id}: {light}: coupling efficiency "
                f"{100 * s.info['coupling_efficiency']:.2f}%, "
                f"transmission {100 * s.info['transmission']:.2f}%"
            )
    if result.readouts:
        lines += ["", "Readouts"]
        for inst_id, values in result.readouts.items():
            parts = ", ".join(f"{k} = {v:.6g}" for k, v in values.items())
            lines.append(f"  {inst_id}: {parts}")
    if result.probes:
        lines += ["", "Probes"]
        for probe_id, values in result.probes.items():
            scalars = {k: v for k, v in values.items() if not isinstance(v, list)}
            parts = ", ".join(f"{k} = {v:.6g}" for k, v in scalars.items())
            lines.append(f"  {probe_id}: {parts}")
    if result.issues:
        lines += ["", "Diagnostics"]
        for issue in result.issues:
            lines.append(f"  [{issue.severity}] {issue.code}: {issue.subject}: {issue.message}")
    return "\n".join(lines)


def _hz(value: float) -> str:
    if value == 0:
        return "0"
    if math.isinf(value):
        return "inf"
    return f"{value:.4g}"


def _ns(value: float) -> str:
    return f"{round(value * 1e9, 3) + 0.0:.4g}"


def _analysis_lines(name: str, a: dict) -> list[str]:
    kind = a["type"]
    if kind == "coincidences":
        return [
            f"  {name} [{kind}] {a['channels'][0]} x {a['channels'][1]}: "
            f"window {_ns(a['window_s'])} ns at {_ns(a['center_s'])} ns "
            f"(peak {_ns(a['peak_s'])} ns), window fraction {a['window_fraction']:.4g}",
            f"      true {_hz(a['true_hz'])} Hz, accidental {_hz(a['accidental_hz'])} Hz, "
            f"CAR = C_true/ACC {_hz(a['car'])} [(C_true+ACC)/ACC = {_hz(a['car_total'])}], "
            f"all-time {_hz(a['coincidence_rate_hz'])} Hz",
            f"      heralding efficiency: {a['heralding_a_given_b']:.4g} (first given second), "
            f"{a['heralding_b_given_a']:.4g} (second given first)",
        ]
    if kind == "heralded_g2":
        return [
            f"  {name} [{kind}] herald {a['herald']}, a {a['a']}, b {a['b']}: "
            f"window {_ns(a['window_s'])} ns",
            f"      R_Ha {_hz(a['rate_ha_hz'])} Hz, R_Hb {_hz(a['rate_hb_hz'])} Hz, "
            f"triples {_hz(a['triples_hz'])} Hz, g2(0) = {a['g2_heralded']:.4g}",
            f"      multimode bunching {a['multimode_bunching']:.4g}, CW bunching "
            f"{a['cw_bunching']:.4g}, herald saturation factor {a['herald_saturation']:.4g}",
        ]
    lines = [
        f"  {name} [{kind}] {a['channels'][0]} x {a['channels'][1]}: "
        f"{len(a['tau_s'])} bins of {_ns(a['bin_width_s'])} ns around {_ns(a['center_s'])} ns, "
        f"peak {_hz(float(max(a['rate_hz'])))} Hz/bin"
    ]
    if "g2_peak" in a:
        lines.append(f"      g2 peak {a['g2_peak']:.4g}, at centre {a['g2_at_center']:.4g}")
    if a.get("g2_per_pulse") is not None:
        lines.append(f"      g2 per pulse {a['g2_per_pulse']:.4g}")
        for role, d in a.get("by_role", {}).items():
            if d["trace_a"] > 0:
                lines.append(
                    f"      {role} arm Schmidt number {d['schmidt_a']:.4g} / {d['schmidt_b']:.4g}"
                )
    return lines


def format_rates_report(result: RateResult) -> str:
    lines = ["Photon rates"]
    if result.sources:
        lines += ["", "Pair sources"]
        for inst_id, s in result.sources.items():
            kind = f"pulsed {s['repetition_rate_hz'] * 1e-6:.4g} MHz" if s["pulsed"] else "CW"
            lines.append(
                f"  {inst_id}: {_hz(s['rate_hz'])} pairs/s ({kind}), dnu {_hz(s['dnu_hz'])} Hz"
            )
    if result.detectors:
        header = (
            f"{'channel':<14}{'detector':<10}{'singles [Hz]':>14}{'pairs [Hz]':>14}"
            f"{'backgr. [Hz]':>14}{'dead [ns]':>11}{'delay [ns]':>12}"
        )
        lines += ["", "Channels", header, "-" * len(header)]
        by_det = {c["detector_instance"]: n for n, c in result.channels.items()}
        for det_id, d in result.detectors.items():
            ch = by_det.get(det_id, "-")
            delay = _ns(result.channels[ch]["delay_s"]) if ch in result.channels else "-"
            lines.append(
                f"{ch:<14}{det_id:<10}{_hz(d['singles_hz']):>14}{_hz(d['pair_singles_hz']):>14}"
                f"{_hz(d['background_hz']):>14}{_ns(d['dead_time_s']):>11}{delay:>12}"
            )
            if d["leakage_hz"]:
                lines.append(f"{'':<24}classical leakage {_hz(d['leakage_hz'])} Hz")
    if result.links:
        lines += ["", "Photon links (photons/s of the pair source(s) at the end of each link)"]
        for link_id, entry in result.links.items():
            parts = [f"{_hz(entry['photons_hz']):>10} photons/s"]
            for role, eta in entry.get("coupling_efficiency", {}).items():
                wl = entry.get("wavelength_m", {}).get(role)
                where = f" at {wl * 1e9:.4g} nm" if wl else ""
                parts.append(f"{role} photon coupling {100 * eta:.2f}%{where}")
            lines.append(f"  {link_id:<12}" + ", ".join(parts))
    if result.pairs:
        lines += ["", "Coincidence rates (all times, after dead time)"]
        for (a, b), p in result.pairs.items():
            lines.append(f"  {a} x {b}: {_hz(p['coincidence_rate_hz'])} Hz")
    if result.analyses:
        lines += ["", "Analyses"]
        for name, a in result.analyses.items():
            lines += _analysis_lines(name, a)
    own = [i for i in result.issues if i not in result.classical.issues]
    if own:
        lines += ["", "Rate diagnostics"]
        for issue in own:
            lines.append(f"  [{issue.severity}] {issue.code}: {issue.subject}: {issue.message}")
    return "\n".join(lines)


def _car(value: float) -> str:
    return "undefined (no accidental estimate)" if math.isnan(value) else f"{value:.4g}"


def format_mc_report(tags, analyses, comparisons=None) -> str:
    lines = [f"Monte Carlo time tags: {tags.duration:g} s, seed {tags.seed}", ""]
    lines.append(f"{'channel':<14}{'events':>12}{'rate [Hz]':>14}")
    for name, n in tags.counts().items():
        lines.append(f"{name:<14}{n:>12d}{n / tags.duration:>14.5g}")
    for name, a in analyses.items():
        lines.append("")
        kind = a["type"]
        if kind == "coincidences":
            lines.append(f"{name} [coincidences] {' x '.join(a['channels'])}")
            lines.append(
                f"  window {a['window_s'] * 1e9:.3g} ns at {a['center_s'] * 1e9:.4g} ns; "
                f"{a['counts']} counts"
            )
            lines.append(
                f"  true {a['true_hz']:.5g} +- {a['true_sigma_hz']:.2g} Hz, accidental "
                f"{a['accidental_hz']:.4g} Hz, CAR "
                f"{_car(a['car'])}"
            )
            if a.get("timing_note"):
                lines.append(f"  note: {a['timing_note']}")
        elif kind == "heralded_g2":
            lines.append(f"{name} [heralded_g2]")
            lines.append(
                f"  triples {a['triples']} ({a['triples_hz']:.4g} Hz), g2(0) = "
                f"{a['g2_heralded']:.4g} +- {a['g2_sigma']:.2g}"
            )
        else:
            lines.append(f"{name} [{kind}] {' x '.join(a['channels'])}")
            lines.append(f"  {len(a['tau_s'])} bins, g2 peak {a['g2_peak']:.4g}")
    if comparisons:
        lines += ["", "Against the rate engine"]
        lines.append(
            f"{'':<2}{'item':<18}{'quantity':<12}{'Monte Carlo':>14}{'engine':>14}{'z':>8}"
        )
        for c in comparisons:
            flag = " " if c.ok else "!"
            lines.append(
                f"{flag} {c.name:<18}{c.quantity:<12}{c.monte_carlo:>14.5g}"
                f"{c.expected:>14.5g}{c.z:>8.2f}"
            )
    return "\n".join(lines)


def format_gaussian_report(result) -> str:
    scale = result.gain_scale
    if isinstance(scale, dict):
        values = set(scale.values())
        scale = (
            f"{values.pop():g}"
            if len(values) == 1
            else ", ".join(f"{k} {v:g}" for k, v in scale.items())
        )
    else:
        scale = f"{scale:g}"
    lines = [
        f"Gaussian multimode statistics per window of {result.window * 1e9:.4g} ns "
        f"(pump scale {scale})",
        "",
    ]
    for name, info in result.sources.items():
        kind = "pulse" if info["pulsed"] else "1/dnu"
        lines.append(
            f"source {name}: {info['pairs_per_window']:.4g} pairs per {kind}, "
            f"{info['modes']} Schmidt modes, {info['members']} member(s)"
        )
    lines += ["", f"{'channel':<14}{'P(click)':>14}{'<n>':>14}{'rate [Hz]':>14}"]
    for ch in result.channels:
        p = result.probability(ch)
        lines.append(
            f"{ch:<14}{p:>14.5g}{result.mean_photons[ch]:>14.5g}{p / result.window:>14.5g}"
        )
    for name, a in result.analyses.items():
        lines.append("")
        if a["type"] == "coincidences":
            lines.append(f"{name} [coincidences] {' x '.join(a['channels'])}")
            lines.append(
                f"  coincidences {a['coincidence_rate_hz']:.5g} Hz, true {a['true_hz']:.5g} Hz, "
                f"accidental {a['accidental_hz']:.4g} Hz, CAR {a['car']:.4g}"
            )
        elif a["type"] == "heralded_g2":
            lines.append(f"{name} [heralded_g2]")
            lines.append(f"  triples {a['triples_hz']:.4g} Hz, g2(0) = {a['g2_heralded']:.4g}")
        else:
            lines.append(f"{name} [g2] {' x '.join(a['channels'])}: g2 = {a['g2_at_center']:.4g}")
    if result.issues:
        lines += ["", "Notes"]
        lines += [f"  {i.code}: {i.message}" for i in result.issues[:8]]
    return "\n".join(lines)


def _t(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9), ("ps", 1e-12)):
        if abs(seconds) >= scale or unit == "ps":
            return f"{seconds / scale:.4g} {unit}"
    return f"{seconds:g} s"


def _approximations(issues, limit: int = 4) -> list[tuple[str, str]]:
    notes = [i for i in issues if i.severity in ("warning", "error")]
    if not notes:
        return [("Approximations", "no engine warnings")]
    text = "; ".join(f"{i.subject}: {i.message}" for i in notes[:limit])
    if len(notes) > limit:
        text += f"; and {len(notes) - limit} more in Diagnostics"
    return [("Approximations", text)]


MC_QUANTITY = {
    "singles_hz": "recorded singles [Hz] (after dead time)",
    "true_hz": "coincidences minus accidentals [Hz]",
    "total_hz": "raw coincidences [Hz]",
    "triples_hz": "heralded triples [Hz]",
}


def gaussian_card(result, calibration: str = "") -> list[tuple[str, str]]:
    """Model summary of a Gaussian-engine result, to show above it."""
    w = result.window
    pulsed = [s.get("pulsed") for s in result.sources.values()]
    if pulsed and all(pulsed):
        window = f"{_t(w)}: one repetition period (pulsed)"
    elif pulsed and not any(pulsed):
        window = f"{_t(w)}: coherence window 1/dnu of the CW source"
    else:
        window = _t(w)
    sources = "; ".join(
        f"{k}: {v['pairs_per_window']:.4g} pairs per window, {v['modes']} Schmidt mode(s), "
        + ("pulsed" if v.get("pulsed") else "CW")
        for k, v in result.sources.items()
    )
    rows = [
        (
            "Model",
            "multimode Gaussian (squeezed-vacuum pairs, thermal per Schmidt mode); exact only "
            "within these Gaussian source and window assumptions",
        ),
        ("Source statistics", sources or "no pair source"),
        ("Statistics window", window),
        (
            "Coincidence",
            "both detectors click in the same statistics window; the analyses' coincidence "
            "windows are not used",
        ),
        ("Timing", "not included: no dead time, jitter, afterpulsing or delays"),
        ("Normalization", "probability per statistics window; rates = probability / window"),
    ]
    if result.pnr_info:
        mass = min(m.get("retained", 1.0) for m in result.pnr_info.values())
        rows.append(("Omitted tail", f"photon-number distributions retain >= {mass:.6g}"))
    else:
        rows.append(("Omitted tail", "none: click probabilities are closed-form"))
    if calibration:
        rows.append(("Calibration", calibration))
    return rows + _approximations(result.issues)


def montecarlo_card(
    tags, analyses, measurement=None, schmidt_number=None, calibration: str = ""
) -> list:
    dead = (
        "detector and tagger dead times set to zero (measurement: no dead time)"
        if measurement is not None and measurement.dead_time == "none"
        else "included: non-paralyzable detector and tagger dead time, jitter, afterpulsing"
    )
    k = schmidt_number
    rows = [
        ("Model", "Monte Carlo time tags of the setup (one random record)"),
        (
            "Source statistics",
            f"thermal pulse statistics of K = {k:g} modes" if k else "Poisson pair emission",
        ),
        ("Timing", dead),
        ("Acquisition", f"{_t(tags.duration)}, seed {tags.seed}"),
    ]
    for name, a in analyses.items():
        if a.get("type") == "coincidences":
            text = (
                f"{_t(a['window_s'])} full width, accidentals from "
                f"{'singles product' if a.get('accidental_model') == 'singles' else 'sidebands'}"
            )
            if a.get("condition"):
                text += f", gated by condition '{a['condition']['name']}'"
            if a.get("timing_note"):
                text += f" ({a['timing_note']})"
            rows.append((f"Coincidence window '{name}'", text))
        elif a.get("type") == "heralded_g2":
            rows.append((f"Coincidence window '{name}'", f"{_t(a['window_s'])} full width"))
    rows.append(
        (
            "Normalization",
            "rates = recorded events / acquisition duration; sigma = Poisson counting of the "
            "recorded events; z = (Monte Carlo - rate engine) / sigma",
        )
    )
    if calibration:
        rows.append(("Calibration", calibration))
    return rows + _approximations(tags.issues)


def sweep_card(result, fock: bool = False, calibration: str = "") -> list:
    failed = len(result.failures)
    rows = [
        ("Model", "every point is a full rate-engine simulation of the edited setup"),
        (
            "Timing",
            "rate engine: independent non-paralyzable dead time per detector, window fractions "
            "from jitter; photon-number (fock:) observables have no dead time",
        ),
        (
            "Normalization",
            "rates in Hz per detector or analysis; fock: probabilities per detection window",
        ),
        (
            "Undefined points",
            f"{failed} point(s) not computed (shown as gaps, listed in the export)"
            if failed
            else "none",
        ),
    ]
    if fock:
        rows.insert(1, ("Photon-number observables", "Photon counts settings at every point"))
    if calibration:
        rows.append(("Calibration", calibration))
    return rows


def chsh_card(result, windows: dict[str, float] | None = None, calibration: str = "") -> list:
    kind = "raw coincidences" if result.quantity == "total_hz" else "accidentals subtracted"
    rows = [
        ("Model", "rate engine at each of the four setting pairs (a forecast, not a measurement)"),
        ("Quantity", f"{kind}; E = (C++ + C-- - C+- - C-+) / (sum of the four)"),
        (
            "Acquisition",
            f"{result.integration_time:g} s per setting pair, 4 pairs "
            f"({4 * result.integration_time:g} s in total)",
        ),
        (
            "Uncertainty",
            "Poisson counting of the forecast counts"
            + (
                f", accidental estimate from {result.sidebands} sideband windows"
                if result.quantity == "true_hz"
                else ""
            )
            + ("" if result.s_sigma is not None else "; undefined: no coincidences predicted"),
        ),
        ("Assumptions", "fair sampling: coincidences post-selected on two detections"),
    ]
    if windows:
        rows.insert(
            2,
            (
                "Coincidence windows",
                ", ".join(f"{n}: {_t(w)} full width" for n, w in windows.items()),
            ),
        )
    if calibration:
        rows.append(("Calibration", calibration))
    return rows


def rates_card(outcome, calibration: str = "") -> list:
    rows = [
        ("Model", "rate engine: mean rates, Poisson multipair accidentals, analytic windows"),
        (
            "Timing",
            "independent non-paralyzable dead time per detector; jitter sets window fractions",
        ),
        ("Normalization", "rates [Hz]; coincidences counted in each analysis' full-width window"),
    ]
    if calibration:
        rows.append(("Calibration", calibration))
    return rows + _approximations(outcome.issues)
