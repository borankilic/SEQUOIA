import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import TypeAdapter, ValidationError
from scipy import constants, sparse
from scipy.special import gammainc

from sequoia.components.base import Component, Context, PortDirection, PortKind
from sequoia.components.detectors import DetectorConfigWarning
from sequoia.data import Duration, ExtrapolationWarning
from sequoia.engines.classical import (
    Network,
    SimulationResult,
    build_network,
    propagate_network,
    simulate,
)
from sequoia.errors import Issue, SetupError
from sequoia.library.store import Library
from sequoia.light.detection import DetectorModel, TaggerModel
from sequoia.light.gaussian import aperture_transmission, overlap
from sequoia.light.pairs import PairEmission
from sequoia.light.state import Beam
from sequoia.setup.model import Setup

C_LIGHT = constants.c
LEAKAGE_FRACTION = 0.01
MAX_FFT = 1 << 22
MAX_UNION = 1 << 18
RESOLUTION = 8.0
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
ROLES = ("signal", "idler")
NULL_STATE = 1e-9
STRONG = 1e-6


class AnalysisSpecError(ValueError):
    pass


@dataclass
class RateResult:
    classical: SimulationResult
    channels: dict[str, dict[str, Any]]
    analyses: dict[str, dict[str, Any]]
    issues: list[Issue]
    detectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    pairs: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)


@dataclass
class _Detector:
    inst: str
    model: DetectorModel
    ports: list[str]


@dataclass
class _Channel:
    name: str
    detector: str
    tagger: str
    port: str
    delay: float
    tagger_model: TaggerModel


@dataclass
class _Source:
    inst: str
    emission: PairEmission
    amp: np.ndarray
    n_i: np.ndarray
    idx: np.ndarray
    rho: dict[str, np.ndarray]
    grid: dict[str, np.ndarray]
    nu0: dict[str, float]
    h: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    singles: dict[str, dict[str, float]] = field(default_factory=dict)
    delays: dict[tuple[str, str], list[tuple[float, float]]] = field(default_factory=dict)
    modes: dict[tuple[str, str], Any] = field(default_factory=dict)
    scale: float = 1.0
    exchange: bool = False
    union: np.ndarray | None = None
    h_union: dict[str, np.ndarray] = field(default_factory=dict)
    rho_union: np.ndarray | None = None
    gram: Any = None

    @property
    def rate(self) -> float:
        return self.emission.source.rate

    @property
    def dnu(self) -> float:
        return self.emission.source.dnu

    @property
    def pulsed(self) -> bool:
        return self.emission.source.repetition_rate is not None


class _Profile:
    """Periodic probability density of the arrival-time difference on a uniform grid."""

    def __init__(self, mass: np.ndarray, dtau: float):
        n = len(mass)
        self.dtau = dtau
        self.period = n * dtau
        self.mass = np.fft.fftshift(mass / mass.sum())
        self.x0 = -self.period / 2.0 - dtau / 2.0
        self.edges = np.arange(n + 1) * dtau
        self.cum = np.concatenate([[0.0], np.cumsum(self.mass)])

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        u = (np.asarray(x, dtype=float) - self.x0) / self.period
        whole = np.floor(u)
        return whole + np.interp((u - whole) * self.period, self.edges, self.cum)

    def window(self, lo: float, hi: float) -> float:
        return float(self.cdf(hi) - self.cdf(lo))

    def density(self, tau: np.ndarray) -> np.ndarray:
        n = len(self.mass)
        k = np.floor((np.asarray(tau, dtype=float) - self.x0) / self.dtau).astype(int) % n
        return self.mass[k] / self.dtau

    def wrap(self, tau: float) -> float:
        return (tau + self.period / 2.0) % self.period - self.period / 2.0


@dataclass
class _Term:
    weight: float
    profile: _Profile
    kind: str
    source: str
    delay: float | None = None
    peak: float | None = None
    shift: float | None = None


def _next_pow2(n: float) -> int:
    return 1 << max(0, math.ceil(math.log2(max(n, 1))))


def _psd_sqrt(e: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(e)
    return (v * np.sqrt(np.clip(w, 0.0, None))[:, None, :]) @ np.conj(np.swapaxes(v, 1, 2))


def _series_survival(rate: float, tau_detector: float, tau_tagger: float) -> float:
    # Poisson arrivals through a non-paralyzable detector (renewal process with interval
    # tau_d + Exp(rate)) followed by a non-paralyzable tagger channel. The tagger accepts the first
    # detector event later than tau_t after its last accepted one, so by Wald's identity the mean
    # accepted interval is (1 + m(tau_t)) * (tau_d + 1/rate), m the renewal function.
    survival_detector = 1.0 / (1.0 + rate * tau_detector)
    if tau_tagger <= tau_detector or tau_tagger <= 0.0 or rate <= 0.0:
        return survival_detector
    if tau_detector <= 0.0:
        return 1.0 / (1.0 + rate * tau_tagger)
    n = int(tau_tagger // tau_detector)
    if n > 10**6:
        renewals = rate * tau_tagger * survival_detector
    else:
        k = np.arange(1, n + 1)
        renewals = float(np.sum(gammainc(k, rate * (tau_tagger - k * tau_detector))))
    return survival_detector / (1.0 + renewals)


def _duration(value: Any, what: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    try:
        return TypeAdapter(Duration).validate_python(value).si
    except ValidationError as exc:
        raise AnalysisSpecError(f"{what}: expected a duration such as '1 ns'") from exc


def _jitter_cf(f: np.ndarray, fwhm: float, fraction: float, tau: float) -> np.ndarray:
    sigma = fwhm * FWHM_TO_SIGMA
    cf = np.exp(-2.0 * np.pi**2 * sigma**2 * f**2).astype(complex)
    if fraction > 0.0 and tau > 0.0:
        cf = cf * ((1.0 - fraction) + fraction / (1.0 + 2j * np.pi * f * tau))
    return cf


def has_rate_devices(setup: Setup, library: Library) -> bool:
    """True when the setup contains a pair source or a single-photon detector."""
    from sequoia.setup.resolve import instantiate

    for inst in setup.instances:
        try:
            comp = instantiate(library, inst)
        except Exception:
            continue
        if type(comp).emit_pairs is not Component.emit_pairs or comp.detector_model() is not None:
            return True
    return False


class _Engine:
    def __init__(self, setup: Setup, library: Library, classical: SimulationResult | None):
        self.setup = setup
        self.library = library
        self.classical = classical if classical is not None else simulate(setup, library)
        self.issues: list[Issue] = []
        self.net: Network = build_network(setup, library)
        self.detectors: dict[str, _Detector] = {}
        self.channels: dict[str, _Channel] = {}
        self.sources: list[_Source] = []
        self.det_data: dict[str, dict[str, Any]] = {}
        self._pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._dtau_hint: dict[tuple[str, str], float] = {}
        self._seen: set[tuple[str, str, str]] = set()

    def issue(self, severity: str, code: str, subject: str, message: str) -> None:
        key = (code, subject, message)
        if key not in self._seen:
            self._seen.add(key)
            self.issues.append(Issue(severity, code, subject, message))  # type: ignore[arg-type]

    def absorb(self, ctx: Context) -> None:
        for i in ctx.issues:
            self.issue(i.severity, i.code, i.subject, i.message)

    def collect_warnings(self, caught: list) -> None:
        for w in caught:
            if issubclass(w.category, ExtrapolationWarning):
                self.issue("warning", "curve_extrapolation", "data", str(w.message))
            elif issubclass(w.category, DetectorConfigWarning):
                self.issue("warning", "detector_config", "data", str(w.message))

    def discover(self) -> None:
        comps = self.net.components
        taggers: dict[str, TaggerModel] = {}
        for inst_id, comp in comps.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = comp.detector_model()
            self.collect_warnings(caught)
            if model is not None:
                ports = [
                    p.name
                    for p in comp.ports()
                    if p.direction == PortDirection.IN and p.kind != PortKind.ELECTRICAL
                ]
                self.detectors[inst_id] = _Detector(inst_id, model, ports)
            try:
                tag = comp.tagger_model()
            except SetupError as exc:
                self.issue("error", "invalid_settings", f"'{inst_id}'", str(exc))
                continue
            if tag is not None:
                taggers[inst_id] = tag
        connected: set[str] = set()
        for link in self.setup.links:
            if link.kind != "cable":
                continue
            src_inst = link.src.split(".")[0]
            dst_inst, dst_port = link.dst.split(".")
            if src_inst in self.detectors and dst_inst in taggers:
                name = link.dst
                delay = link.delay.si + taggers[dst_inst].input_delays.get(dst_port, 0.0)
                self.channels[name] = _Channel(
                    name, src_inst, dst_inst, dst_port, delay, taggers[dst_inst]
                )
                connected.add(src_inst)
        for inst_id in self.detectors:
            if inst_id not in connected:
                self.issue(
                    "warning",
                    "no_tagger_connection",
                    f"'{inst_id}'",
                    "detector output is not connected to a time tagger channel; its counts "
                    "are reported but cannot be used in coincidence analyses",
                )

    def port_light(self, inst_id: str, port: str):
        for state in self.classical.links.values():
            if state.dst == f"{inst_id}.{port}":
                return state.end
        return None

    def emit_sources(self) -> None:
        ctx = Context(issues=[])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for inst_id in self.net.order:
                comp = self.net.components[inst_id]
                lights = {}
                for p in comp.ports():
                    light = self.port_light(inst_id, p.name)
                    if light is not None:
                        lights[p.name] = light
                ctx.subject = f"'{inst_id}'"
                ctx.instance_id = inst_id
                emission = comp.emit_pairs(lights, ctx)
                if emission is not None:
                    self.sources.append(self.prepare_source(inst_id, emission))
        self.absorb(ctx)
        self.collect_warnings(caught)

    def prepare_source(self, inst_id: str, emission: PairEmission) -> _Source:
        ps = emission.source
        amp = np.array(ps.amplitude, dtype=complex)
        n_s = np.asarray(ps.n_signal, dtype=np.int64)
        n_sum = np.asarray(ps.n_sum, dtype=np.int64)
        n_i_all = n_sum[None, :] - n_s[:, None]
        valid = (n_i_all > 0) & (n_s[:, None] > 0)
        lost = float(np.sum(np.abs(amp[..., ~valid]) ** 2))
        if lost > 1e-9:
            raise SetupError(
                f"pair source '{inst_id}': {100 * lost:.3g}% of the pair amplitude lies at "
                "non-positive frequencies"
            )
        amp[..., ~valid] = 0.0
        if np.any(np.diff(n_s) <= 0):
            raise SetupError(f"pair source '{inst_id}': n_signal must be strictly increasing")
        n_i = np.unique(n_i_all[valid])
        idx = np.searchsorted(n_i, np.where(valid, n_i_all, n_i[0]))
        rho_s = np.einsum("sijm,tijm->jst", amp, np.conj(amp))
        r_i = np.einsum("sijm,skjm->jmik", amp, np.conj(amp))
        flat = idx.ravel()
        rho_i = np.zeros((len(n_i), 2, 2), dtype=complex)
        for a in range(2):
            for b in range(2):
                v = r_i[..., a, b].ravel()
                rho_i[:, a, b] = np.bincount(flat, weights=v.real, minlength=len(n_i))
                rho_i[:, a, b] += 1j * np.bincount(flat, weights=v.imag, minlength=len(n_i))
        weight_s = np.real(np.trace(rho_s, axis1=1, axis2=2))
        weight_i = np.real(np.trace(rho_i, axis1=1, axis2=2))
        nu0 = {
            "signal": float(np.sum(n_s * weight_s) * ps.dnu),
            "idler": float(np.sum(n_i * weight_i) * ps.dnu),
        }
        src = _Source(
            inst_id,
            emission,
            amp,
            n_i,
            idx,
            {"signal": rho_s, "idler": rho_i},
            {"signal": n_s, "idler": n_i},
            nu0,
        )
        if emission.signal_port == emission.idler_port:
            self.setup_exchange(src, n_s, n_sum)
        return src

    def setup_exchange(self, src: _Source, n_s: np.ndarray, n_sum: np.ndarray) -> None:
        amp = src.amp
        partner = n_sum[None, :] - n_s[:, None]
        pos = np.clip(np.searchsorted(n_s, partner), 0, len(n_s) - 1)
        found = n_s[pos] == partner
        overlap_sum = 0.0
        for m in range(len(n_sum)):
            js = np.flatnonzero(found[:, m])
            if len(js) == 0:
                continue
            direct = amp[:, :, js, m]
            swapped = amp[:, :, pos[js, m], m].transpose(1, 0, 2)
            overlap_sum += float(np.real(np.sum(np.conj(direct) * swapped)))
        if abs(overlap_sum) < 1e-12:
            return
        src.exchange = True
        norm = 1.0 + overlap_sum
        if norm < NULL_STATE:
            src.scale = 0.0
            self.issue(
                "warning",
                "bosonic_null_state",
                f"'{src.inst}'",
                "signal and idler share one spatial mode and the pair amplitude is "
                "antisymmetric under exchange of the two photons; the symmetrised two-photon "
                "state vanishes and no pairs are emitted",
            )
        else:
            src.scale = 1.0 / math.sqrt(norm)
        ps = src.emission.source
        if abs(overlap(ps.signal_mode, ps.idler_mode)) < 0.99 * math.sqrt(
            abs(overlap(ps.signal_mode, ps.signal_mode) * overlap(ps.idler_mode, ps.idler_mode))
        ):
            self.issue(
                "warning",
                "same_port_mode_mismatch",
                f"'{src.inst}'",
                "signal and idler leave the same port with different spatial modes; the "
                "exchange normalisation assumes identical spatial modes",
            )
        union = np.union1d(n_s, src.n_i)
        u = len(union)
        pos_s = np.searchsorted(union, n_s)
        pos_i = np.searchsorted(union, src.n_i)
        s, i, j, m = np.nonzero(amp)
        f = sparse.coo_matrix(
            (amp[s, i, j, m], (s * u + pos_s[j], i * u + pos_i[src.idx[j, m]])),
            shape=(2 * u, 2 * u),
        ).tocsr()
        src.gram = (f + f.T).tocsr()
        src.union = union

    def transfer_sources(self) -> None:
        for src in self.sources:
            ps = src.emission.source
            arrivals: dict[str, dict[tuple[str, str], list[Beam]]] = {}
            for role in ROLES:
                grid = src.grid[role]
                port = src.emission.signal_port if role == "signal" else src.emission.idler_port
                mode = ps.signal_mode if role == "signal" else ps.idler_mode
                wavelengths = C_LIGHT / (grid * ps.dnu)
                n = len(grid)
                seed = Beam(
                    f"{src.inst}:{role}",
                    wavelengths,
                    np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy(),
                    0.0,
                    0.0,
                    mode,
                    0.0,
                    (src.inst,),
                )

                def emit(inst_id, comp, ctx, seed=seed, port=port, owner=src.inst):
                    return [(port, seed)] if inst_id == owner else []

                ctx = Context(issues=[])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    arrived, _ = propagate_network(
                        self.net, ctx, emit, coupling_note=f"{role} photon "
                    )
                self.absorb(ctx)
                self.collect_warnings(caught)
                arrivals[role] = arrived
            for det_id, det in self.detectors.items():
                src.h[det_id] = {}
                radius_of = getattr(self.net.components[det_id], "aperture_radius", None)
                radius = radius_of() if radius_of is not None else None
                for role in ROLES:
                    grid = src.grid[role]
                    nu = grid * ps.dnu
                    beams = [b for p in det.ports for b in arrivals[role].get((det_id, p), [])]
                    t = np.zeros((len(grid), 2, 2), dtype=complex)
                    nu0 = src.nu0[role]
                    strengths = []
                    for b in beams:
                        phase = np.exp(2j * np.pi * ((nu - nu0) * b.gpl + nu0 * b.opl) / C_LIGHT)
                        collected = 1.0
                        if radius is not None:
                            collected = aperture_transmission(radius, b.mode)
                            phase = phase * math.sqrt(collected)
                        t += b.jones * phase[:, None, None]
                        strengths.append(float(np.max(np.abs(b.jones) ** 2)) * collected)
                    self.check_modes(det_id, role, beams)
                    src.delays[(det_id, role)] = [
                        (b.gpl / C_LIGHT, w)
                        for b, w in zip(beams, strengths, strict=True)
                        if w > STRONG
                    ]
                    src.modes[(det_id, role)] = None
                    if strengths and max(strengths) > STRONG:
                        src.modes[(det_id, role)] = beams[int(np.argmax(strengths))].mode
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        eff = np.asarray(det.model.efficiency(C_LIGHT / nu), dtype=complex)
                    self.collect_warnings(caught)
                    src.h[det_id][role] = _psd_sqrt(eff) @ t
                self.check_role_modes(src, det_id)
                if src.exchange:
                    src.singles[det_id] = {
                        "signal": self.exchange_singles(src, det_id),
                        "idler": 0.0,
                    }
                    continue
                src.singles[det_id] = {
                    role: src.rate
                    * float(
                        np.sum(
                            np.einsum(
                                "npb,nbc,npc->n",
                                src.h[det_id][role],
                                src.rho[role],
                                np.conj(src.h[det_id][role]),
                            ).real
                        )
                    )
                    for role in ROLES
                }

    def exchange_singles(self, src: _Source, det_id: str) -> float:
        ps = src.emission.source
        n_s = np.asarray(ps.n_signal, dtype=np.int64)
        union = src.union
        h = np.zeros((len(union), 2, 2), dtype=complex)
        h[np.searchsorted(union, src.n_i)] = src.h[det_id]["idler"]
        h[np.searchsorted(union, n_s)] = src.h[det_id]["signal"]
        src.h_union[det_id] = h
        hm = sparse.bmat(
            [[sparse.diags(h[:, p, b]) for b in range(2)] for p in range(2)], format="csr"
        )
        y = hm @ src.gram
        return src.rate * src.scale**2 * float(np.sum(np.abs(y.data) ** 2))

    def exchange_rho(self, src: _Source) -> np.ndarray:
        if src.rho_union is None:
            u = len(src.union)
            ggt = (src.gram @ src.gram.conj().T).tocsr()
            k = np.arange(u)
            rho = np.zeros((u, 2, 2), dtype=complex)
            for p in range(2):
                for q in range(2):
                    rho[:, p, q] = np.asarray(ggt[p * u + k, q * u + k]).ravel()
            src.rho_union = rho * src.scale**2
        return src.rho_union

    def check_role_modes(self, src: _Source, det_id: str) -> None:
        s, i = src.modes.get((det_id, "signal")), src.modes.get((det_id, "idler"))
        if s is None or i is None:
            return
        c = self.mode_overlap(s, i)
        if abs(c) < 0.99:
            self.issue(
                "warning",
                "photon_mode_mismatch",
                f"'{det_id}'",
                f"signal and idler photons reach the detector in spatial modes with overlap "
                f"{abs(c) ** 2:.3g}; two-photon interference between them is reduced by that "
                "factor",
            )

    @staticmethod
    def mode_overlap(p, q) -> complex:
        den = math.sqrt(abs(overlap(p, p)) * abs(overlap(q, q)))
        return complex(np.conj(overlap(p, q)) / den) if den > 0 else 0.0

    def mode_weight(self, src: _Source, a: str, b: str) -> complex:
        sa, ia = src.modes.get((a, "signal")), src.modes.get((a, "idler"))
        sb, ib = src.modes.get((b, "signal")), src.modes.get((b, "idler"))
        if sa is None or ia is None or sb is None or ib is None:
            return 1.0
        return self.mode_overlap(sa, ia) * np.conj(self.mode_overlap(sb, ib))

    def check_modes(self, det_id: str, role: str, beams: list[Beam]) -> None:
        strong = [b for b in beams if float(np.max(np.abs(b.jones) ** 2)) > 1e-6]
        for i, p in enumerate(strong):
            for q in strong[i + 1 :]:
                num = abs(overlap(p.mode, q.mode))
                den = math.sqrt(abs(overlap(p.mode, p.mode)) * abs(overlap(q.mode, q.mode)))
                if den > 0 and num / den < 0.99:
                    self.issue(
                        "warning",
                        "path_mode_mismatch",
                        f"'{det_id}'",
                        f"{role} photons reach the detector through paths whose spatial "
                        f"modes overlap only {100 * num / den:.1f}%; the rate engine adds "
                        "the paths coherently, so interference visibility is overestimated",
                    )
                    return

    def background(self) -> None:
        for det_id, det in self.detectors.items():
            model = det.model
            leak = 0.0
            radius_of = getattr(self.net.components[det_id], "aperture_radius", None)
            radius = radius_of() if radius_of is not None else None
            for port in det.ports:
                light = self.port_light(det_id, port)
                if light is None:
                    continue
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    eff = np.asarray(model.efficiency(light.wavelengths), dtype=complex)
                self.collect_warnings(caught)
                energy = constants.h * C_LIGHT / light.wavelengths
                rate = np.einsum("nab,nba->n", eff, light.coherency).real / energy
                collected = 1.0 if radius is None else aperture_transmission(radius, light.mode)
                leak += float(np.sum(rate)) * collected
            pair = {s.inst: sum(s.singles[det_id].values()) for s in self.sources}
            pair_total = sum(pair.values())
            dark = model.dark_count_rate
            after = model.afterpulse_probability * (pair_total + dark + leak)
            incident_bg = dark + leak + after
            incident = pair_total + incident_bg
            tagger_dead = 0.0
            for ch in self.channels.values():
                if ch.detector == det_id:
                    tagger_dead = max(tagger_dead, ch.tagger_model.dead_time)
            dead = max(model.dead_time, tagger_dead)
            survival = _series_survival(incident, model.dead_time, tagger_dead)
            effective = (1.0 / survival - 1.0) / incident if incident > 0 else dead
            self.det_data[det_id] = {
                "detector": model.kind,
                "singles_hz": incident * survival,
                "pair_singles_hz": pair_total * survival,
                "background_hz": incident_bg * survival,
                "incident_hz": incident,
                "incident_pair_hz": pair_total,
                "incident_background_hz": incident_bg,
                "dark_hz": dark,
                "leakage_hz": leak,
                "afterpulse_hz": after,
                "dead_time_s": dead,
                "dead_time_detector_s": model.dead_time,
                "dead_time_tagger_s": tagger_dead,
                "dead_time_effective_s": effective,
                "survival": survival,
                "pair_by_source": pair,
            }
            if incident > 0 and leak > LEAKAGE_FRACTION * incident:
                self.issue(
                    "warning",
                    "pump_leakage",
                    f"'{det_id}'",
                    f"classical light reaching the detector produces {leak:.4g} counts/s, "
                    f"{100 * leak / incident:.1f}% of its singles",
                )

    def channel_lookup(self, name: str, analysis: str) -> _Channel:
        if not isinstance(name, str) or name not in self.channels:
            known = ", ".join(sorted(self.channels)) or "none"
            raise AnalysisSpecError(
                f"analysis '{analysis}': unknown channel '{name}' (connected channels: {known})"
            )
        return self.channels[name]

    def amplitude_blocks(self, src: _Source, a: str, b: str) -> list[np.ndarray]:
        ha, hb = src.h[a], src.h[b]
        amp, idx = src.amp, src.idx
        ps = src.emission.source
        n_s = np.asarray(ps.n_signal, dtype=np.int64)
        n_sum = np.asarray(ps.n_sum, dtype=np.int64)
        phi1 = np.einsum("jps,jmqi,sijm->pqjm", ha["signal"], hb["idler"][idx], amp, optimize=True)
        phi2 = np.einsum("jmpi,jqs,sijm->pqjm", ha["idler"][idx], hb["signal"], amp, optimize=True)
        use1, use2 = bool(np.any(phi1)), bool(np.any(phi2))
        if not (use1 or use2):
            return []
        support = np.sum(np.abs(amp) ** 2, axis=(0, 1)) > 0
        if not support.any():
            return []
        js = np.flatnonzero(support.any(axis=1))
        n_a2 = n_sum[None, :] - n_s[:, None]
        m = len(n_sum)
        mg = np.broadcast_to(np.arange(m)[None, :], n_a2.shape)
        scale = src.scale
        kappa = self.mode_weight(src, a, b) if use1 and use2 else 1.0
        parts = []
        if use1:
            parts.append((n_s[js], None, scale * phi1[:, :, js, :]))
        if use2:
            parts.append((n_a2[support], mg[support], scale * kappa * phi2[:, :, support]))
        lo = min(int(p.min()) for p, _, _ in parts)
        hi = max(int(p.max()) for p, _, _ in parts)
        split = hi - lo + 1 > MAX_UNION
        groups = [[p] for p in parts] if split else [parts]
        if split and len(parts) == 2:
            (p1, _, _), (p2, _, _) = parts
            if p1.min() <= p2.max() and p2.min() <= p1.max():
                self.issue(
                    "warning",
                    "interference_dropped",
                    f"'{src.inst}'",
                    "the frequency range of the pair amplitude is too large to hold both photon "
                    "assignments coherently and the two overlap; their interference is "
                    "dropped. Reduce the number of lattice points of the pair source",
                )
        if use1 and use2 and abs(kappa) < 1.0 - 1e-12:
            rest = math.sqrt(max(0.0, 1.0 - abs(kappa) ** 2))
            groups.append([(n_a2[support], mg[support], scale * rest * phi2[:, :, support])])
        blocks = []
        for group in groups:
            g_lo = min(int(p.min()) for p, _, _ in group)
            g_hi = max(int(p.max()) for p, _, _ in group)
            arr = np.zeros((2, 2, g_hi - g_lo + 1, m), dtype=complex)
            for pos, mi, values in group:
                if mi is None:
                    arr[:, :, pos - g_lo, :] += values
                else:
                    arr[:, :, pos - g_lo, mi] += values
            blocks.append(arr)
        return blocks

    def kernel_cf(self, chan_a: _Channel, chan_b: _Channel, f: np.ndarray) -> np.ndarray:
        ma, mb = self.detectors[chan_a.detector].model, self.detectors[chan_b.detector].model
        cf = _jitter_cf(f, ma.jitter_fwhm, ma.jitter_tail_fraction, ma.jitter_tail_tau)
        cf = cf * np.conj(
            _jitter_cf(f, mb.jitter_fwhm, mb.jitter_tail_fraction, mb.jitter_tail_tau)
        )
        for ch in (chan_a, chan_b):
            cf = cf * _jitter_cf(f, ch.tagger_model.jitter_fwhm, 0.0, 0.0)
        return cf

    def dtau_target(self, chan_a: _Channel, chan_b: _Channel, window: float | None) -> float:
        scales = []
        for ch in (chan_a, chan_b):
            m = self.detectors[ch.detector].model
            scales += [m.jitter_fwhm, ch.tagger_model.jitter_fwhm]
            if m.jitter_tail_fraction > 0:
                scales.append(m.jitter_tail_tau)
        if window is not None:
            scales.append(window)
        scales = [s for s in scales if s > 0]
        return min(scales) / RESOLUTION if scales else math.inf

    def choose_nfft(self, length: int, period: float, dtau_target: float, subject: str) -> int:
        n_min = _next_pow2(2 * length)
        n_res = _next_pow2(math.ceil(period / dtau_target)) if math.isfinite(dtau_target) else 0
        n = max(n_min, n_res)
        if n > MAX_FFT:
            n = max(n_min, MAX_FFT)
            self.issue(
                "warning",
                "timing_resolution",
                subject,
                f"time grid of {period / n * 1e12:.3g} ps is coarser than a fraction of the "
                "shortest jitter or window; reduce the frequency lattice spacing of the source "
                "or enlarge the window for accurate window fractions",
            )
        return n

    def pair_profile(
        self, chan_a: _Channel, chan_b: _Channel, window: float | None
    ) -> dict[str, Any]:
        key = (chan_a.name, chan_b.name)
        target = self.dtau_target(chan_a, chan_b, window)
        cached = self._pair_cache.get(key)
        if cached is not None and cached["target"] <= target * (1 + 1e-9):
            return cached
        terms: list[_Term] = []
        total = 0.0
        subject = f"{chan_a.name} / {chan_b.name}"
        for src in self.sources:
            period = 1.0 / src.dnu
            blocks = self.amplitude_blocks(src, chan_a.detector, chan_b.detector)
            if not blocks:
                continue
            rate_all = src.rate * sum(float(np.sum(np.abs(b) ** 2)) for b in blocks)
            self.check_delays(src, chan_a, chan_b, period, subject)
            length = max(b.shape[2] for b in blocks)
            n = self.choose_nfft(length, period, target, subject)
            g = np.zeros(n)
            chunk = max(1, int(4e6 // n))
            for arr in blocks:
                for p in range(2):
                    for q in range(2):
                        for start in range(0, arr.shape[3], chunk):
                            x = np.fft.fft(arr[p, q, :, start : start + chunk], n=n, axis=0)
                            g += np.sum(np.abs(x) ** 2, axis=1) / n
            dtau = period / n
            g = self.apply_kernel(g, dtau, chan_a, chan_b)
            if rate_all > 0:
                delay = self.strongest_delay(src, chan_a.detector, chan_b.detector)
                terms.append(_Term(rate_all, _Profile(g, dtau), "pair", src.inst, delay))
            total += rate_all
        result = {"target": target, "rate_all": total, "terms": terms}
        self._pair_cache[key] = result
        return result

    def check_delays(self, src, a: _Channel, b: _Channel, period: float, subject: str) -> None:
        da, db = a.detector, b.detector
        delays = [
            x - y
            for ra, rb in (("signal", "idler"), ("idler", "signal"))
            for x, _ in src.delays.get((da, ra), [])
            for y, _ in src.delays.get((db, rb), [])
        ]
        if delays and max(delays) - min(delays) > period / 2.0:
            self.issue(
                "warning",
                "delay_aliasing",
                subject,
                f"group-delay differences between photon paths span "
                f"{(max(delays) - min(delays)) * 1e9:.3g} ns, more than half the period "
                f"1/dnu = {period * 1e9:.3g} ns of the source frequency lattice; correlation "
                "peaks alias. Use a finer lattice spacing in the pair source",
            )
        elif delays and max(abs(d) for d in delays) > period / 2.0:
            self.issue(
                "warning",
                "delay_aliasing",
                subject,
                f"the group-delay difference between the photon paths reaches "
                f"{max(abs(d) for d in delays) * 1e9:.3g} ns, more than half the period "
                f"1/dnu = {period * 1e9:.3g} ns of the source frequency lattice; the peak "
                "position is unwrapped from the path delays, but the lattice is too coarse "
                "to resolve other paths without aliasing. Use a finer lattice spacing in "
                "the pair source",
            )

    def apply_kernel(self, g: np.ndarray, dtau: float, a: _Channel, b: _Channel) -> np.ndarray:
        f = np.fft.rfftfreq(len(g), dtau)
        spectrum = np.fft.rfft(g) * self.kernel_cf(a, b, f)
        return np.clip(np.fft.irfft(spectrum, n=len(g)), 0.0, None)

    def thermal_terms(
        self, chan_a: _Channel, chan_b: _Channel, window: float | None
    ) -> list[_Term]:
        terms = []
        target = self.dtau_target(chan_a, chan_b, window)
        subject = f"{chan_a.name} / {chan_b.name}"
        for src in self.sources:
            if src.pulsed:
                continue
            if src.exchange:
                items = [
                    (
                        src.union,
                        src.h_union[chan_a.detector],
                        src.h_union[chan_b.detector],
                        self.exchange_rho(src),
                    )
                ]
            else:
                items = [
                    (src.grid[r], src.h[chan_a.detector][r], src.h[chan_b.detector][r], src.rho[r])
                    for r in ROLES
                ]
            for grid, ha, hb, rho in items:
                c = src.rate * np.einsum("npb,nbc,npc->n", ha, rho, np.conj(hb))
                if not np.any(np.abs(c) > 0):
                    continue
                pos = grid - grid.min()
                arr = np.zeros(int(pos.max()) + 1, dtype=complex)
                arr[pos] = c
                period = 1.0 / src.dnu
                n = self.choose_nfft(len(arr), period, target, subject)
                g1 = np.fft.fft(arr, n=n)
                power = np.abs(g1) ** 2
                dtau = period / n
                weight = float(np.sum(power) * dtau)
                if weight <= 0:
                    continue
                power = self.apply_kernel(power, dtau, chan_a, chan_b)
                delay = self.strongest_delay(src, chan_a.detector, chan_b.detector, same_role=True)
                terms.append(_Term(weight, _Profile(power, dtau), "thermal", src.inst, delay))
        return terms

    def survival(self, chan: _Channel) -> float:
        return self.det_data[chan.detector]["survival"]

    def delay_offset(self, a: _Channel, b: _Channel) -> float:
        return a.delay - b.delay

    def strongest_delay(
        self, src: _Source, a: str, b: str, same_role: bool = False
    ) -> float | None:
        pairs = (("signal", "signal"), ("idler", "idler"))
        if not same_role:
            pairs = (("signal", "idler"), ("idler", "signal"))
        best, weight = None, 0.0
        for ra, rb in pairs:
            for x, sx in src.delays.get((a, ra), []):
                for y, sy in src.delays.get((b, rb), []):
                    if sx * sy > weight:
                        best, weight = x - y, sx * sy
        return best

    def term_shift(self, term: _Term) -> float:
        if term.shift is None:
            term.peak = self.find_peak([term])
            period = term.profile.period
            term.shift = 0.0
            if term.delay is not None:
                term.shift = round((term.delay - term.peak) / period) * period
        return term.shift

    def unwrapped_peak(self, terms: list[_Term]) -> float:
        terms = [t for t in terms if t.weight > 0]
        if not terms:
            return 0.0
        strongest = max(terms, key=lambda t: t.weight)
        return self.find_peak(terms) + self.term_shift(strongest)

    def window_fraction(self, term: _Term, center: float, window: float) -> float:
        period = term.profile.period
        if window >= period:
            return 1.0
        peak = term.peak if term.peak is not None else 0.0
        lo = max(center - window / 2.0, peak - period / 2.0)
        hi = min(center + window / 2.0, peak + period / 2.0)
        if hi <= lo:
            return 0.0
        return term.profile.window(lo, hi)

    def find_peak(self, terms: list[_Term]) -> float:
        terms = [t for t in terms if t.weight > 0]
        if not terms:
            return 0.0
        finest = min(t.profile.dtau for t in terms)
        span = min(t.profile.period for t in terms)
        n = int(min(span / finest, 1 << 21))
        step = span / n
        tau = (np.arange(n) - n // 2) * step
        dens = sum(t.weight * t.profile.density(tau) for t in terms)
        k = int(np.argmax(dens))
        if 0 < k < n - 1:
            y0, y1, y2 = dens[k - 1], dens[k], dens[k + 1]
            denom = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            return float(tau[k] + shift * step)
        return float(tau[k])

    def coincidence_numbers(
        self, a: _Channel, b: _Channel, window: float, center: float | str
    ) -> dict[str, Any]:
        data = self.pair_profile(a, b, window)
        terms: list[_Term] = data["terms"]
        offset = self.delay_offset(a, b)
        peak = self.unwrapped_peak(terms)
        center_opt = peak if isinstance(center, str) else center - offset
        s_a, s_b = self.survival(a), self.survival(b)
        fractions = {}
        in_window = 0.0
        for t in terms:
            frac = self.window_fraction(t, center_opt - self.term_shift(t), window)
            fractions[t.source] = frac
            in_window += t.weight * frac
        rate_all = data["rate_all"]
        fraction = min(in_window / rate_all, 1.0) if rate_all > 0 else 0.0
        in_window = fraction * rate_all
        if terms and window >= min(t.profile.period for t in terms):
            self.issue(
                "warning",
                "window_exceeds_period",
                f"{a.name} / {b.name}",
                f"the coincidence window ({window * 1e9:.3g} ns) is not shorter than the "
                f"period 1/dnu ({min(t.profile.period for t in terms) * 1e9:.3g} ns) of the "
                "source frequency lattice; the window fraction saturates at 1",
            )
        da, db = self.det_data[a.detector], self.det_data[b.detector]
        sa, sb = da["singles_hz"], db["singles_hz"]
        acc = window * sa * sb
        for src in self.sources:
            if not src.pulsed:
                continue
            period = 1.0 / src.emission.source.repetition_rate
            if window >= period:
                continue
            pa = s_a * da["pair_by_source"][src.inst]
            pb = s_b * db["pair_by_source"][src.inst]
            acc += pa * pb * (period - window)
        true = in_window * s_a * s_b
        return {
            "rate_all": rate_all * s_a * s_b,
            "true": true,
            "accidental": acc,
            "fraction": fraction,
            "center_opt": center_opt,
            "peak_opt": peak,
            "peak_measured": peak + offset,
            "center_measured": center_opt + offset,
            "singles_a": sa,
            "singles_b": sb,
        }

    def parse_center(self, spec: dict[str, Any], name: str) -> float | str:
        center = spec.get("center", "auto")
        if isinstance(center, str) and center.strip().lower() == "auto":
            return "auto"
        return _duration(center, f"analysis '{name}': center")

    def analyse(self, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = str(spec.get("name", spec.get("type", "analysis")))
        kind = spec.get("type")
        handler = {
            "coincidences": self.a_coincidences,
            "heralded_g2": self.a_heralded,
            "g2": self.a_histogram,
            "histogram": self.a_histogram,
        }.get(kind)
        if handler is None:
            raise AnalysisSpecError(f"analysis '{name}': unknown type {kind!r}")
        return name, handler(name, spec)

    def two_channels(self, name: str, spec: dict[str, Any]) -> tuple[_Channel, _Channel]:
        chans = spec.get("channels")
        if not isinstance(chans, list) or len(chans) != 2:
            raise AnalysisSpecError(f"analysis '{name}': 'channels' must list two channel names")
        a = self.channel_lookup(chans[0], name)
        b = self.channel_lookup(chans[1], name)
        if a.name == b.name:
            raise AnalysisSpecError(f"analysis '{name}': the two channels must differ")
        return a, b

    def window_flag(self, name: str, numbers: dict[str, Any], window: float, subject: str) -> None:
        peak = numbers["peak_opt"]
        center = numbers["center_opt"]
        if numbers["rate_all"] > 0 and abs(peak - center) > window / 2.0:
            self.issue(
                "warning",
                "coincidence_outside_window",
                f"analysis '{name}'",
                f"the correlation peak lies {abs(peak - center) * 1e9:.3g} ns from the window "
                f"centre; only {100 * numbers['fraction']:.1f}% of the pair coincidences fall "
                "inside the window",
            )

    def a_coincidences(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        a, b = self.two_channels(name, spec)
        window = _duration(spec.get("window"), f"analysis '{name}': window")
        if window <= 0:
            raise AnalysisSpecError(f"analysis '{name}': window must be positive")
        center = self.parse_center(spec, name)
        n = self.coincidence_numbers(a, b, window, center)
        self.window_flag(name, n, window, name)
        acc = n["accidental"]
        true = n["true"]
        return {
            "type": "coincidences",
            "channels": [a.name, b.name],
            "window_s": window,
            "center_s": n["center_measured"],
            "peak_s": n["peak_measured"],
            "window_fraction": n["fraction"],
            "coincidence_rate_hz": n["rate_all"],
            "true_hz": true,
            "accidental_hz": acc,
            "total_hz": true + acc,
            "car": true / acc if acc > 0 else math.inf,
            "singles_a_hz": n["singles_a"],
            "singles_b_hz": n["singles_b"],
            "heralding_a_given_b": true / n["singles_b"] if n["singles_b"] > 0 else 0.0,
            "heralding_b_given_a": true / n["singles_a"] if n["singles_a"] > 0 else 0.0,
        }

    def uncorrelated_probability(self, chan: _Channel, window: float) -> float:
        d = self.det_data[chan.detector]
        surv = d["survival"]
        probability, covered = 0.0, 0.0
        for src in self.sources:
            if not src.pulsed:
                continue
            rep = src.emission.source.repetition_rate
            if window >= 1.0 / rep:
                continue
            pair = surv * d["pair_by_source"][src.inst]
            probability += pair / rep
            covered += pair
        return probability + max(d["singles_hz"] - covered, 0.0) * window

    def a_heralded(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        herald = self.channel_lookup(spec.get("herald"), name)
        a = self.channel_lookup(spec.get("a"), name)
        b = self.channel_lookup(spec.get("b"), name)
        window = _duration(spec.get("window"), f"analysis '{name}': window")
        center = self.parse_center(spec, name)
        n_a = self.coincidence_numbers(herald, a, window, center)
        n_b = self.coincidence_numbers(herald, b, window, center)
        self.window_flag(name + ".a", n_a, window, name)
        self.window_flag(name + ".b", n_b, window, name)
        s_h = self.det_data[herald.detector]["singles_hz"]
        p_a = self.uncorrelated_probability(a, window)
        p_b = self.uncorrelated_probability(b, window)
        triples = n_a["true"] * p_b + n_b["true"] * p_a + s_h * p_a * p_b
        meas_a = n_a["true"] + n_a["accidental"]
        meas_b = n_b["true"] + n_b["accidental"]
        g2 = triples * s_h / (meas_a * meas_b) if meas_a > 0 and meas_b > 0 else math.nan
        return {
            "type": "heralded_g2",
            "herald": herald.name,
            "a": a.name,
            "b": b.name,
            "window_s": window,
            "herald_singles_hz": s_h,
            "rate_ha_hz": meas_a,
            "rate_hb_hz": meas_b,
            "true_ha_hz": n_a["true"],
            "true_hb_hz": n_b["true"],
            "triples_hz": triples,
            "g2_heralded": g2,
        }

    def bin_masses(self, terms: list[_Term], scale: float, edges: np.ndarray, offset: float):
        out = np.zeros(len(edges) - 1)
        for t in terms:
            cdf = t.profile.cdf(edges - offset - self.term_shift(t))
            out += t.weight * scale * np.diff(cdf)
        return out

    def a_histogram(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        a, b = self.two_channels(name, spec)
        rng = _duration(spec.get("range"), f"analysis '{name}': range")
        try:
            bins = int(spec.get("bins", 200))
        except (TypeError, ValueError) as exc:
            raise AnalysisSpecError(f"analysis '{name}': bins must be an integer") from exc
        if rng <= 0 or bins < 1:
            raise AnalysisSpecError(f"analysis '{name}': range and bins must be positive")
        bin_w = rng / bins
        for src in self.sources:
            if rng > 1.0 / src.dnu:
                self.issue(
                    "warning",
                    "range_exceeds_period",
                    f"analysis '{name}'",
                    f"the histogram range exceeds the period 1/dnu = {1e9 / src.dnu:.3g} ns "
                    "of the source frequency lattice; the correlation repeats inside the range",
                )
        data = self.pair_profile(a, b, bin_w)
        thermal = self.thermal_terms(a, b, bin_w)
        offset = self.delay_offset(a, b)
        pair_terms: list[_Term] = data["terms"]
        center = self.parse_center(spec, name)
        if isinstance(center, str):
            peak_terms = pair_terms + thermal
            if not any(t.weight > 0 for t in peak_terms):
                center_opt = 0.0
            else:
                center_opt = self.unwrapped_peak(peak_terms)
            center_meas = center_opt + offset
        else:
            center_meas = center
        edges = center_meas + (np.arange(bins + 1) - bins / 2.0) * bin_w
        s_a, s_b = self.survival(a), self.survival(b)
        shift = offset
        true = self.bin_masses(pair_terms, s_a * s_b, edges, shift)
        excess = self.bin_masses(thermal, s_a * s_b, edges, shift)
        da, db = self.det_data[a.detector], self.det_data[b.detector]
        floor = da["singles_hz"] * db["singles_hz"] * bin_w * np.ones(bins)
        rate = true + excess + floor
        centers = 0.5 * (edges[:-1] + edges[1:])
        out: dict[str, Any] = {
            "type": spec["type"],
            "channels": [a.name, b.name],
            "tau_s": centers,
            "bin_width_s": bin_w,
            "center_s": center_meas,
            "true_hz": true,
            "thermal_hz": excess,
            "accidental_hz": floor,
            "rate_hz": rate,
        }
        pulsed = [s for s in self.sources if s.pulsed]
        if spec["type"] == "g2":
            out["g2"] = rate / floor if np.all(floor > 0) else np.full(bins, math.nan)
            out["g2_peak"] = float(np.max(out["g2"]))
            middle = (
                out["g2"][bins // 2 - 1 : bins // 2 + 1] if bins % 2 == 0 else out["g2"][bins // 2]
            )
            out["g2_at_center"] = float(np.mean(middle))
        if pulsed:
            out.update(self.pulsed_g2(a, b))
            out["note"] = (
                "pulsed source: the accidental floor is the average uncorrelated rate "
                "S_a*S_b spread uniformly; per-pulse g2 is in 'g2_per_pulse'"
            )
        elif spec["type"] == "g2":
            out["g2_per_pulse"] = None
        return out

    def pulsed_g2(self, a: _Channel, b: _Channel) -> dict[str, Any]:
        pulsed = [s for s in self.sources if s.pulsed]
        src = pulsed[0]
        if len(pulsed) > 1:
            self.issue(
                "warning",
                "multiple_pulsed_sources",
                f"{a.name} / {b.name}",
                "per-pulse g2 is computed for the first pulsed source only",
            )
        if src.exchange:
            self.issue(
                "warning",
                "same_mode_g2_approximation",
                f"{a.name} / {b.name}",
                "per-pulse g2 and Schmidt numbers of a source whose signal and idler share one "
                "spatial mode treat the two roles separately and neglect their exchange terms",
            )
        da, db = self.det_data[a.detector], self.det_data[b.detector]
        f_rep = src.emission.source.repetition_rate
        mu = src.rate / f_rep
        pa, pb = da["incident_hz"] / f_rep, db["incident_hz"] / f_rep
        excess = 0.0
        by_role: dict[str, dict[str, float]] = {}
        for role in ROLES:
            x = self.partner_matrix(src, role)
            ya = np.einsum("fps,fsk->fpk", src.h[a.detector][role], x).reshape(-1, x.shape[2])
            yb = np.einsum("fps,fsk->fpk", src.h[b.detector][role], x).reshape(-1, x.shape[2])
            ga, gb = np.conj(ya).T @ ya, np.conj(yb).T @ yb
            cross = float(np.real(np.sum(ga * gb.T)))
            tr_a, tr_b = float(np.real(np.trace(ga))), float(np.real(np.trace(gb)))
            excess += mu**2 * cross
            by_role[role] = {
                "trace_a": tr_a,
                "trace_b": tr_b,
                "schmidt_a": tr_a**2 / float(np.real(np.sum(ga * ga.T))) if tr_a > 0 else math.nan,
                "schmidt_b": tr_b**2 / float(np.real(np.sum(gb * gb.T))) if tr_b > 0 else math.nan,
            }
        pair = sum(t.weight for t in self.pair_profile(a, b, None)["terms"] if t.source == src.inst)
        denominator = pa * pb
        g2 = (pair / f_rep + denominator + excess) / denominator if denominator > 0 else math.nan
        return {"g2_per_pulse": g2, "by_role": by_role, "pulsed_source": src.inst}

    def partner_matrix(self, src: _Source, role: str) -> np.ndarray:
        amp, idx = src.amp, src.idx
        ns = amp.shape[2]
        ni = len(src.n_i)
        jg = np.broadcast_to(np.arange(ns)[:, None], idx.shape)
        if role == "signal":
            x = np.zeros((ns, 2, 2 * ni), dtype=complex)
            for k in range(2):
                np.add.at(x, (jg, slice(None), k * ni + idx), amp[:, k].transpose(1, 2, 0))
            return x
        x = np.zeros((ni, 2, 2 * ns), dtype=complex)
        for k in range(2):
            np.add.at(x, (idx, slice(None), k * ns + jg), amp[k].transpose(1, 2, 0))
        return x

    def pair_rate(self, a: _Channel, b: _Channel) -> float:
        total = 0.0
        for src in self.sources:
            blocks = self.amplitude_blocks(src, a.detector, b.detector)
            total += src.rate * sum(float(np.sum(np.abs(blk) ** 2)) for blk in blocks)
        return total * self.survival(a) * self.survival(b)


def simulate_rates(
    setup: Setup, library: Library, classical: SimulationResult | None = None
) -> RateResult:
    engine = _Engine(setup, library, classical)
    engine.discover()
    engine.emit_sources()
    engine.transfer_sources()
    engine.background()
    analyses: dict[str, dict[str, Any]] = {}
    for spec in setup.analyses:
        try:
            name, result = engine.analyse(spec)
        except AnalysisSpecError as exc:
            engine.issue("error", "bad_analysis", "analyses", str(exc))
            continue
        analyses[name] = result
    channels = {}
    for name, ch in engine.channels.items():
        channels[name] = {
            **engine.det_data[ch.detector],
            "detector_instance": ch.detector,
            "tagger": ch.tagger,
            "delay_s": ch.delay,
        }
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    names = list(engine.channels)
    if len(names) <= 8 and engine.sources:
        for i, x in enumerate(names):
            for y in names[i + 1 :]:
                rate = engine.pair_rate(engine.channels[x], engine.channels[y])
                pairs[(x, y)] = {"coincidence_rate_hz": rate}
    sources = {
        s.inst: {
            "rate_hz": s.rate,
            "dnu_hz": s.dnu,
            "pulsed": s.pulsed,
            "repetition_rate_hz": s.emission.source.repetition_rate,
        }
        for s in engine.sources
    }
    if not engine.sources and engine.detectors:
        engine.issue(
            "warning",
            "no_pair_source",
            "setup",
            "no component emitted photon pairs; only classical leakage and dark counts are "
            "reported",
        )
    issues = list(engine.classical.issues)
    issues += [i for i in engine.issues if i not in issues]
    return RateResult(engine.classical, channels, analyses, issues, engine.det_data, sources, pairs)
