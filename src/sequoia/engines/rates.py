import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import TypeAdapter, ValidationError
from scipy import constants, sparse
from scipy.special import gammainc

from sequoia.components.base import Component, Context, PortDirection, PortKind
from sequoia.components.detectors import DetectorConfigWarning
from sequoia.data import Duration, ExtrapolationWarning
from sequoia.engines.cache import (
    DEFAULT_MEMORY_LIMIT,
    Cancelled,
    ResourceLimitError,
    RunCache,
    check_cancel,
    check_memory,
    emit_with_cache,
    record_warnings,
    run_library,
)
from sequoia.engines.classical import (
    LinkState,
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
ALIAS_STRENGTH = 1e-3
DEAD_TIME_LOSS_WARNING = 0.1
MIN_PAIR_FRACTION = 1e-6
GRAM_SPACE: str | None = None
MAX_SIMILARITY_MEMBERS = 16
__all__ = ["Cancelled", "ResourceLimitError", "RunCache"]


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
    links: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    members: np.ndarray
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


def _min_eigenvalue(e: np.ndarray) -> np.ndarray:
    trace = (e[:, 0, 0] + e[:, 1, 1]).real
    det = (e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]).real
    return 0.5 * (trace - np.sqrt(np.clip(trace**2 - 4.0 * det, 0.0, None)))


def _psd_sqrt(e: np.ndarray) -> np.ndarray:
    # Closed form for a 2x2 positive semidefinite matrix: sqrt(A) = (A + s I) / sqrt(tr A + 2 s)
    # with s = sqrt(det A); entries with a negative eigenvalue are clipped through eigh.
    e = np.asarray(e, dtype=complex)
    trace = (e[:, 0, 0] + e[:, 1, 1]).real
    det = (e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]).real
    bad = _min_eigenvalue(e) < -1e-12 * np.maximum(trace, 1e-300)
    s = np.sqrt(np.clip(det, 0.0, None))
    tau = np.sqrt(np.clip(trace + 2.0 * s, 0.0, None))
    safe = np.where(tau > 0, tau, 1.0)
    out = (e + s[:, None, None] * np.eye(2)) / safe[:, None, None]
    out[tau <= 0] = 0.0
    if np.any(bad):
        w, v = np.linalg.eigh(e[bad])
        out[bad] = (v * np.sqrt(np.clip(w, 0.0, None))[:, None, :]) @ np.conj(np.swapaxes(v, 1, 2))
    return out


def _quadratic_trace(h: np.ndarray, rho: np.ndarray) -> float:
    # sum_n tr(h_n rho_n h_n^dagger)
    return float(np.sum(((h @ rho) * np.conj(h)).real))


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


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return math.nan if numerator <= 0 else math.inf


def _jitter_cf(f: np.ndarray, fwhm: float, fraction: float, tau: float) -> np.ndarray:
    sigma = fwhm * FWHM_TO_SIGMA
    cf = np.exp(-2.0 * np.pi**2 * sigma**2 * f**2).astype(complex)
    if fraction > 0.0 and tau > 0.0:
        cf = cf * ((1.0 - fraction) + fraction / (1.0 + 2j * np.pi * f * tau))
    return cf


def has_rate_devices(
    setup: Setup,
    library: Library,
    cache: RunCache | None = None,
    network: Network | None = None,
) -> bool:
    """True when the setup contains a pair source or a single-photon detector."""
    from sequoia.setup.resolve import instantiate

    if network is not None:
        comps = list(network.components.values())
    else:
        library = run_library(library, cache)
        comps = []
        for inst in setup.instances:
            try:
                comps.append(instantiate(library, inst))
            except Exception:
                continue
    for comp in comps:
        if type(comp).emit_pairs is not Component.emit_pairs or comp.detector_model() is not None:
            return True
    return False


class _Engine:
    def __init__(
        self,
        setup: Setup,
        library: Library,
        classical: SimulationResult | None,
        should_cancel: Callable[[], bool] | None = None,
        cache: RunCache | None = None,
        max_bytes: float | None = None,
    ):
        self.setup = setup
        self.cache = cache
        self.library = run_library(library, cache)
        self.should_cancel = should_cancel
        self.max_bytes = DEFAULT_MEMORY_LIMIT if max_bytes is None else float(max_bytes)
        if classical is None:
            classical = simulate(setup, library, should_cancel=should_cancel, cache=cache)
        self.classical = classical
        self.issues: list[Issue] = []
        network = getattr(classical, "network", None)
        if network is not None and network.setup is setup:
            self.net: Network = network
        else:
            self.net = build_network(setup, self.library)
        self.link_photons: dict[str, dict[str, Any]] = {}
        self._thermal_cache: dict[tuple, list] = {}
        self.detectors: dict[str, _Detector] = {}
        self.channels: dict[str, _Channel] = {}
        self.sources: list[_Source] = []
        self.det_data: dict[str, dict[str, Any]] = {}
        self._pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._dtau_hint: dict[tuple[str, str], float] = {}
        self._seen: set[tuple[str, str, str]] = set()
        self._moments: dict[tuple[str, str, str], dict[str, Any]] = {}

    def cancel_point(self) -> None:
        check_cancel(self.should_cancel)

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
            self.cancel_point()
            with record_warnings() as caught:
                model = comp.detector_model()
            self.collect_warnings(caught)
            if model is not None:
                for field_name in ("dark_count_rate", "dead_time", "jitter_fwhm"):
                    value = getattr(model, field_name)
                    if not math.isfinite(value) or value < 0:
                        raise SetupError(
                            f"detector '{inst_id}': {field_name} = {value} is not a finite "
                            "non-negative number"
                        )
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
        for inst_id in self.net.order:
            self.cancel_point()
            comp = self.net.components[inst_id]
            lights = {}
            for p in comp.ports():
                light = self.port_light(inst_id, p.name)
                if light is not None:
                    lights[p.name] = light
            ctx.subject = f"'{inst_id}'"
            ctx.instance_id = inst_id
            emission, caught = emit_with_cache(self.cache, comp, lights, ctx)
            self.collect_warnings(caught)
            if emission is not None:
                self.sources.append(self.prepare_source(inst_id, emission))
        self.absorb(ctx)

    def prepare_source(self, inst_id: str, emission: PairEmission) -> _Source:
        ps = emission.source
        n_s = np.asarray(ps.n_signal, dtype=np.int64)
        n_sum = np.asarray(ps.n_sum, dtype=np.int64)
        count = len(ps.members) * len(n_s) * len(n_sum)
        check_memory(
            count * (2 * 64 + 120),
            self.max_bytes,
            f"pair source '{inst_id}' ({len(n_s)} signal x {len(n_sum)} sum-frequency lattice "
            f"points, {len(ps.members)} member(s))",
            "use a coarser frequency lattice (larger dnu or fewer samples) in the pair source",
        )
        members = np.array(ps.members, dtype=complex)
        n_i_all = n_sum[None, :] - n_s[:, None]
        valid = (n_i_all > 0) & (n_s[:, None] > 0)
        lost = float(np.sum(np.abs(members[..., ~valid]) ** 2))
        if lost > 1e-9:
            raise SetupError(
                f"pair source '{inst_id}': {100 * lost:.3g}% of the pair amplitude lies at "
                "non-positive frequencies"
            )
        members[..., ~valid] = 0.0
        if np.any(np.diff(n_s) <= 0):
            raise SetupError(f"pair source '{inst_id}': n_signal must be strictly increasing")
        n_i = np.unique(n_i_all[valid])
        idx = np.searchsorted(n_i, np.where(valid, n_i_all, n_i[0]))
        rho_s = np.zeros((len(n_s), 2, 2), dtype=complex)
        rho_i = np.zeros((len(n_i), 2, 2), dtype=complex)
        for amp in members:
            if np.any(amp):
                rs, ri = self.member_rho(amp, idx, len(n_i))
                rho_s += rs
                rho_i += ri
        weight_s = np.real(np.trace(rho_s, axis1=1, axis2=2))
        weight_i = np.real(np.trace(rho_i, axis1=1, axis2=2))
        nu0 = {
            "signal": float(np.sum(n_s * weight_s) * ps.dnu),
            "idler": float(np.sum(n_i * weight_i) * ps.dnu),
        }
        src = _Source(
            inst_id,
            emission,
            members,
            n_i,
            idx,
            {"signal": rho_s, "idler": rho_i},
            {"signal": n_s, "idler": n_i},
            nu0,
        )
        if emission.signal_port == emission.idler_port:
            if len(members) > 1:
                raise SetupError(
                    f"pair source '{inst_id}': a mixed pair state (ensemble) leaving one port "
                    "for both photons is not supported; use separate signal and idler ports"
                )
            self.setup_exchange(src, n_s, n_sum)
        return src

    @staticmethod
    def member_rho(amp: np.ndarray, idx: np.ndarray, n_idler: int) -> tuple[np.ndarray, np.ndarray]:
        rho_s = np.einsum("sijm,tijm->jst", amp, np.conj(amp))
        r_i = np.einsum("sijm,skjm->jmik", amp, np.conj(amp))
        flat = idx.ravel()
        rho_i = np.zeros((n_idler, 2, 2), dtype=complex)
        for a in range(2):
            for b in range(2):
                v = r_i[..., a, b].ravel()
                rho_i[:, a, b] = np.bincount(flat, weights=v.real, minlength=n_idler)
                rho_i[:, a, b] += 1j * np.bincount(flat, weights=v.imag, minlength=n_idler)
        return rho_s, rho_i

    def setup_exchange(self, src: _Source, n_s: np.ndarray, n_sum: np.ndarray) -> None:
        amp = src.members[0]
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

    def record_link_photons(self, src: _Source, role: str, ends, states) -> None:
        rho = src.rho[role]
        for link_id, beams in ends.items():
            flux = 0.0
            for beam in beams:
                j = beam.jones
                if len(j) != len(rho):
                    continue
                flux += _quadratic_trace(j, rho)
            flux *= src.rate
            if flux <= 0.0:
                continue
            entry = self.link_photons.setdefault(
                link_id, {"by_source": {}, "coupling_efficiency": {}, "transmission": {}}
            )
            entry["by_source"].setdefault(src.inst, {})[role] = flux
            info = states[link_id].info
            if "coupling_efficiency" in info:
                entry["coupling_efficiency"][role] = info["coupling_efficiency"]
                entry["transmission"][role] = info["transmission"]
            entry["wavelength_m"] = entry.get("wavelength_m", {})
            entry["wavelength_m"][role] = C_LIGHT / src.nu0[role]

    def transfer_sources(self) -> None:
        for src in self.sources:
            self.cancel_point()
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
                states = {
                    link.id: LinkState(link.id, link.kind, link.src, link.dst, None)
                    for link in self.net.setup.links
                }
                with record_warnings() as caught:
                    arrived, ends = propagate_network(
                        self.net,
                        ctx,
                        emit,
                        states=states,
                        coupling_note=f"{role} photon ",
                        should_cancel=self.should_cancel,
                    )
                self.absorb(ctx)
                self.collect_warnings(caught)
                arrivals[role] = arrived
                self.record_link_photons(src, role, ends, states)
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
                    with record_warnings() as caught:
                        eff = np.asarray(det.model.efficiency(C_LIGHT / nu), dtype=complex)
                    self.collect_warnings(caught)
                    self.validate_efficiency(det_id, eff)
                    src.h[det_id][role] = _psd_sqrt(eff) @ t
                self.check_role_modes(src, det_id)
                if src.exchange:
                    src.singles[det_id] = {
                        "signal": self.exchange_singles(src, det_id),
                        "idler": 0.0,
                    }
                    continue
                src.singles[det_id] = {
                    role: src.rate * _quadratic_trace(src.h[det_id][role], src.rho[role])
                    for role in ROLES
                }

    def validate_efficiency(self, det_id: str, eff: np.ndarray) -> None:
        if not np.all(np.isfinite(eff)):
            raise SetupError(
                f"detector '{det_id}': the efficiency is not finite (NaN or inf) over the "
                "photon spectrum; check the detector efficiency curve and parameters"
            )
        low = float(np.min(_min_eigenvalue(eff))) if len(eff) else 0.0
        if low < -1e-9:
            self.issue(
                "warning",
                "detector_efficiency_not_psd",
                f"'{det_id}'",
                f"the efficiency operator has a negative eigenvalue ({low:.3g}); negative parts "
                "are clipped to zero",
            )

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
                with record_warnings() as caught:
                    eff = np.asarray(model.efficiency(light.wavelengths), dtype=complex)
                self.collect_warnings(caught)
                self.validate_efficiency(det_id, eff)
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
            self.saturation_issues(det_id, incident, leak, survival, dead, model.dead_time)
            if incident > 0 and leak > LEAKAGE_FRACTION * incident:
                self.issue(
                    "warning",
                    "pump_leakage",
                    f"'{det_id}'",
                    f"classical light reaching the detector produces {leak:.4g} counts/s, "
                    f"{100 * leak / incident:.1f}% of its singles",
                )

    def source_diagnostics(self) -> None:
        if not self.detectors:
            return
        for src in self.sources:
            if src.rate <= 0:
                continue
            best = max(sum(src.singles[d].values()) for d in self.detectors) / src.rate
            if best < MIN_PAIR_FRACTION:
                signal = C_LIGHT / src.nu0["signal"] * 1e9 if src.nu0["signal"] > 0 else math.nan
                idler = C_LIGHT / src.nu0["idler"] * 1e9 if src.nu0["idler"] > 0 else math.nan
                self.issue(
                    "warning",
                    "no_phase_matched_pairs",
                    f"'{src.inst}'",
                    f"pair source '{src.inst}' emits {src.rate:.4g} pairs/s but at most "
                    f"{best:.2g} detected photons per pair (below {MIN_PAIR_FRACTION:g}) reach any "
                    f"detector; the pair spectrum (signal about {signal:.1f} nm, idler about "
                    f"{idler:.1f} nm) lies outside the passbands of the filters or detectors on "
                    "the photon paths, or the crystal is detuned from phase matching for this "
                    "pump (temperature, poling period, pump wavelength)",
                )

    def saturation_issues(
        self, det_id: str, incident: float, leak: float, survival: float, dead: float, own: float
    ) -> None:
        loss = 1.0 - survival
        if loss > DEAD_TIME_LOSS_WARNING:
            limit = (
                f"; the counts cannot exceed 1/dead time = {1 / dead:.4g} Hz" if dead > 0 else ""
            )
            self.issue(
                "warning",
                "detector_saturation",
                f"'{det_id}'",
                f"{incident:.4g} counts/s arrive at the detector and its dead time of "
                f"{dead * 1e9:.4g} ns loses {100 * loss:.1f}% of them (recorded singles "
                f"{incident * survival:.4g} Hz){limit}; coincidences, CAR and heralding "
                "efficiencies are reduced accordingly",
            )
        if leak > 0 and dead > 0:
            tagger = dead if dead > own else 0.0
            leak_loss = 1.0 - _series_survival(leak, own, tagger)
            if leak_loss > DEAD_TIME_LOSS_WARNING:
                self.issue(
                    "warning",
                    "leakage_saturation",
                    f"'{det_id}'",
                    f"classical light alone produces {leak:.4g} counts/s, which already loses "
                    f"{100 * leak_loss:.1f}% of the counts to the {dead * 1e9:.4g} ns dead time "
                    "before any photon pair is detected; block or attenuate the classical light",
                )

    def channel_lookup(self, name: str, analysis: str) -> _Channel:
        if not isinstance(name, str) or name not in self.channels:
            known = ", ".join(sorted(self.channels)) or "none"
            raise AnalysisSpecError(
                f"analysis '{analysis}': unknown channel '{name}' (connected channels: {known})"
            )
        return self.channels[name]

    def amplitude_blocks(self, src: _Source, a: str, b: str) -> list[np.ndarray]:
        blocks: list[np.ndarray] = []
        for amp in src.members:
            if np.any(amp):
                blocks += self.member_blocks(src, amp, a, b)
        return blocks

    def member_blocks(self, src: _Source, amp: np.ndarray, a: str, b: str) -> list[np.ndarray]:
        ha, hb = src.h[a], src.h[b]
        idx = src.idx
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
            check_memory(
                64.0 * (g_hi - g_lo + 1) * m,
                self.max_bytes,
                f"the time-correlation amplitudes of pair source '{src.inst}' "
                f"({g_hi - g_lo + 1} frequency lattice points x {m} sum frequencies)",
                "use a coarser frequency lattice or fewer samples in the pair source",
            )
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
        check_memory(
            n * 48.0,
            self.max_bytes,
            f"the FFT of {n} points for the time correlation of {subject}",
            "use a coarser frequency lattice (larger dnu) or fewer samples in the pair source",
        )
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
                            self.cancel_point()
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
        combos = [
            (x - y, sx * sy)
            for ra, rb in (("signal", "idler"), ("idler", "signal"))
            for x, sx in src.delays.get((da, ra), [])
            for y, sy in src.delays.get((db, rb), [])
        ]
        if len(combos) < 2:
            return
        top = max(w for _, w in combos)
        delays = [d for d, w in combos if w >= ALIAS_STRENGTH * top]
        if len(delays) > 1 and max(delays) - min(delays) > period / 2.0:
            self.issue(
                "warning",
                "delay_aliasing",
                subject,
                f"group-delay differences between photon paths that interfere span "
                f"{(max(delays) - min(delays)) * 1e9:.3g} ns, more than half the period "
                f"1/dnu = {period * 1e9:.3g} ns of the source frequency lattice; the "
                "correlation peaks of the paths alias onto each other. Use a finer lattice "
                "spacing in the pair source",
            )

    def apply_kernel(self, g: np.ndarray, dtau: float, a: _Channel, b: _Channel) -> np.ndarray:
        f = np.fft.rfftfreq(len(g), dtau)
        spectrum = np.fft.rfft(g) * self.kernel_cf(a, b, f)
        return np.clip(np.fft.irfft(spectrum, n=len(g)), 0.0, None)

    def thermal_terms(
        self, chan_a: _Channel, chan_b: _Channel, window: float | None
    ) -> list[_Term]:
        target = self.dtau_target(chan_a, chan_b, window)
        cache_key = (chan_a.name, chan_b.name, target)
        if cache_key in self._thermal_cache:
            return self._thermal_cache[cache_key]
        terms: list[_Term] = []
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
                c = src.rate * np.einsum("npb,nbc,nqc->npq", ha, rho, np.conj(hb))
                if not np.any(np.abs(c) > 0):
                    continue
                pos = grid - grid.min()
                check_memory(
                    128.0 * (int(pos.max()) + 1),
                    self.max_bytes,
                    f"the thermal correlation of {subject} "
                    f"({int(pos.max()) + 1} frequency lattice points)",
                    "use a coarser frequency lattice or fewer samples in the pair source",
                )
                arr = np.zeros((int(pos.max()) + 1, 2, 2), dtype=complex)
                arr[pos] = c
                period = 1.0 / src.dnu
                n = self.choose_nfft(len(arr), period, target, subject)
                g1 = np.fft.fft(arr, n=n, axis=0)
                power = np.sum(np.abs(g1) ** 2, axis=(1, 2))
                dtau = period / n
                weight = float(np.sum(power) * dtau)
                if weight <= 0:
                    continue
                power = self.apply_kernel(power, dtau, chan_a, chan_b)
                delay = self.strongest_delay(src, chan_a.detector, chan_b.detector, same_role=True)
                terms.append(_Term(weight, _Profile(power, dtau), "thermal", src.inst, delay))
        self._thermal_cache[cache_key] = terms
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
            pa = s_a * da["pair_by_source"][src.inst]
            pb = s_b * db["pair_by_source"][src.inst]
            if pa * pb <= 0.0:
                continue
            count = self.same_pulse_count(src, a, b, window, center_opt, period)
            acc += pa * pb * (period * count - window)
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

    def uncorrelated_delay(self, src: _Source, a: str, b: str) -> float:
        best = []
        for det in (a, b):
            singles = src.singles.get(det, {})
            role = max(ROLES, key=lambda r: singles.get(r, 0.0))
            paths = src.delays.get((det, role), [])
            best.append(max(paths, key=lambda p: p[1])[0] if paths else 0.0)
        return best[0] - best[1]

    def same_pulse_count(
        self, src: _Source, a: _Channel, b: _Channel, window: float, centre: float, period: float
    ) -> float:
        # Expected number of pulse offsets k for which the arrival-time difference of two
        # uncorrelated photons of pulses k periods apart falls in the window: the jitter
        # distribution of the difference folded onto one repetition period and integrated over
        # the window (a staircase for sharp timing, 1 for a window inside one period).
        target = self.dtau_target(a, b, window)
        n = _next_pow2(max(period / target if math.isfinite(target) else 0, 4096))
        n = min(n, MAX_FFT)
        delta = np.zeros(n)
        delta[0] = 1.0
        dtau = period / n
        profile = _Profile(self.apply_kernel(delta, dtau, a, b), dtau)
        d = centre - self.uncorrelated_delay(src, a.detector, b.detector)
        return profile.window(d - window / 2.0, d + window / 2.0)

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
            "car": _ratio(true, acc),
            "car_total": _ratio(true + acc, acc),
            "singles_a_hz": n["singles_a"],
            "singles_b_hz": n["singles_b"],
            "heralding_a_given_b": true / n["singles_b"] if n["singles_b"] > 0 else 0.0,
            "heralding_b_given_a": true / n["singles_a"] if n["singles_a"] > 0 else 0.0,
        }

    def uncorrelated_probability(self, chan: _Channel, window: float) -> tuple[float, float]:
        d = self.det_data[chan.detector]
        surv = d["survival"]
        probability, covered, per_pulse = 0.0, 0.0, 0.0
        for src in self.sources:
            if not src.pulsed:
                continue
            rep = src.emission.source.repetition_rate
            if window >= 1.0 / rep:
                continue
            pair = surv * d["pair_by_source"][src.inst]
            probability += pair / rep
            per_pulse += pair / rep
            covered += pair
        return probability + max(d["singles_hz"] - covered, 0.0) * window, per_pulse

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
        p_a, q_a = self.uncorrelated_probability(a, window)
        p_b, q_b = self.uncorrelated_probability(b, window)
        subject = f"{herald.name} / {a.name} / {b.name}"
        factor, bunching, saturation = 1.0, 0.0, 1.0
        pulsed = [s for s in self.sources if s.pulsed]
        if pulsed and q_a > 0 and q_b > 0:
            moments = self.pulsed_moments(pulsed[0], a, b, herald)
            self.ensemble_model_issue(pulsed[0], moments, subject)
            factor = moments["pair_pair_factor"]
            bunching = moments["bunching_conditional"]
            saturation = moments["herald_saturation"]
        b1, b2, qc_a, qc_b = self.cw_bunching(a, b, n_a, n_b, window)
        for src in self.sources:
            if not src.pulsed and src.emission.source.is_ensemble:
                self.cw_ensemble_similarity(src, a, b)
        triples = (
            n_a["true"] * (p_b + (factor - 1.0) * q_b + b1 * qc_b)
            + n_b["true"] * (p_a + (factor - 1.0) * q_a + b1 * qc_a)
            + s_h * (p_a * p_b + b2 * qc_a * qc_b)
        )
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
            "multimode_bunching": bunching,
            "herald_saturation": saturation,
            "cw_bunching": b1,
            "cw_bunching_accidental": b2,
            "g2_heralded": g2,
        }

    def cw_bunching(self, a: _Channel, b: _Channel, n_a, n_b, window: float):
        # Thermal bunching of the a and b photons of continuous sources: the excess coincidence
        # density between a and b (jitter-convolved |G1|^2 profile) integrated over the b window,
        # relative to the uncorrelated product of the pair-sourced singles, and the same integral
        # with the triangular kernel of two free windows for the accidental triples.
        cw = [s for s in self.sources if not s.pulsed]
        da, db = self.det_data[a.detector], self.det_data[b.detector]
        inc_a = sum(da["pair_by_source"][s.inst] for s in cw)
        inc_b = sum(db["pair_by_source"][s.inst] for s in cw)
        q_a, q_b = self.survival(a) * inc_a * window, self.survival(b) * inc_b * window
        terms = self.thermal_terms(a, b, window) if cw else []
        product = sum(da["pair_by_source"][s.inst] * db["pair_by_source"][s.inst] for s in cw)
        if not terms or product <= 0 or window <= 0:
            return 0.0, 0.0, q_a, q_b
        centre = n_b["center_opt"] - n_a["center_opt"]
        mass = 0.0
        for t in terms:
            shift = self.term_shift(t)
            mass += t.weight * self.window_fraction(t, centre - shift, window)
        u = np.linspace(0.0, window, 401)
        profile = np.zeros_like(u)
        for t in terms:
            peak = t.peak if t.peak is not None else 0.0
            half = t.profile.period / 2.0
            c = centre - self.term_shift(t)
            lo = np.clip(c - u, peak - half, peak + half)
            hi = np.clip(c + u, peak - half, peak + half)
            profile += t.weight * (t.profile.cdf(hi) - t.profile.cdf(lo))
        integral = float(np.sum(0.5 * (profile[1:] + profile[:-1]) * np.diff(u)))
        return mass / (product * window), integral / (product * window**2), q_a, q_b

    def bin_masses(self, terms: list[_Term], scale: float, edges: np.ndarray, offset: float):
        out = np.zeros(len(edges) - 1)
        for t in terms:
            x = edges - offset - self.term_shift(t)
            peak = t.peak if t.peak is not None else 0.0
            half = t.profile.period / 2.0
            out += t.weight * scale * np.diff(t.profile.cdf(np.clip(x, peak - half, peak + half)))
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
                    "of the source frequency lattice; only one copy of the correlation "
                    "(the principal period around the peak) is shown and the periodic images "
                    "of the lattice are suppressed",
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
            try:
                out.update(self.pulsed_g2(a, b))
            except ResourceLimitError as exc:
                self.issue(
                    "warning",
                    "memory_limit",
                    f"analysis '{name}'",
                    f"{exc}; the per-pulse g2 and Schmidt numbers are not computed",
                )
                out["g2_per_pulse"] = None
            out["note"] = (
                "pulsed source: the accidental floor is the average uncorrelated rate "
                "S_a*S_b spread uniformly; per-pulse g2 is in 'g2_per_pulse'"
            )
        elif spec["type"] == "g2":
            out["g2_per_pulse"] = None
        if spec["type"] == "g2":
            for src in self.sources:
                if not src.pulsed and src.emission.source.is_ensemble:
                    self.cw_ensemble_similarity(src, a, b)
        return out

    def pulsed_g2(self, a: _Channel, b: _Channel) -> dict[str, Any]:
        pulsed = [s for s in self.sources if s.pulsed]
        src = pulsed[0]
        subject = f"{a.name} / {b.name}"
        if len(pulsed) > 1:
            self.issue(
                "warning",
                "multiple_pulsed_sources",
                subject,
                "per-pulse g2 is computed for the first pulsed source only",
            )
        if src.exchange:
            self.issue(
                "warning",
                "same_mode_g2_approximation",
                subject,
                "per-pulse g2 and Schmidt numbers of a source whose signal and idler share one "
                "spatial mode treat the two roles separately and neglect their exchange terms",
            )
        da, db = self.det_data[a.detector], self.det_data[b.detector]
        f_rep = src.emission.source.repetition_rate
        mu = src.rate / f_rep
        pa, pb = da["incident_hz"] / f_rep, db["incident_hz"] / f_rep
        moments = self.pulsed_moments(src, a, b)
        self.ensemble_model_issue(src, moments, subject)
        excess = mu**2 * moments["excess"]
        pair = sum(t.weight for t in self.pair_profile(a, b, None)["terms"] if t.source == src.inst)
        denominator = pa * pb
        g2 = (pair / f_rep + denominator + excess) / denominator if denominator > 0 else math.nan
        return {"g2_per_pulse": g2, "by_role": moments["by_role"], "pulsed_source": src.inst}

    def ensemble_model_issue(self, src: _Source, moments: dict[str, Any], subject: str) -> None:
        similar = moments.get("similar")
        if similar is None or similar:
            return
        self.issue(
            "warning",
            "ensemble_g2_model",
            f"'{src.inst}'",
            f"the mixed pair state of '{src.inst}' has members with different spectral "
            f"structure (smallest normalised overlap of their reduced one-photon operators "
            f"{moments['min_overlap']:.3f}), so its second-order statistics depend on the "
            "mixing model. Pulsed sources: every pulse is entirely in one member, so g2 is the "
            "per pulse average of 1 + 1/K_k over members, whereas the Schmidt number reported "
            "is that of the averaged one-body density matrix; CW sources: independent pairs "
            "are drawn from the mixture (one-body density matrix) and g2 follows from its "
            "purity. 1 + 1/K is not the g2 here, and a pure multimode JSA must not be "
            "encoded as an ensemble of its Schmidt modes",
        )

    def cw_ensemble_similarity(self, src: _Source, a: _Channel, b: _Channel) -> None:
        members = src.members
        if len(members) < 2 or len(members) > MAX_SIMILARITY_MEMBERS:
            return
        weights = src.emission.source.member_weights
        active = [k for k in range(len(members)) if weights[k] > 0]
        n_i = len(src.n_i)
        overlap_min, spread = 1.0, 0.0
        for det in (a.detector, b.detector):
            for role in ROLES:
                h = src.h[det][role]
                ops = []
                for k in active:
                    rs, ri = self.member_rho(members[k], src.idx, n_i)
                    rho = rs if role == "signal" else ri
                    ops.append(np.einsum("npb,nbc,nqc->npq", h, rho, np.conj(h)) / weights[k])
                norms = [float(np.sum(np.abs(o) ** 2)) for o in ops]
                traces = [float(np.real(np.trace(o, axis1=1, axis2=2).sum())) for o in ops]
                if max(traces) <= 0:
                    continue
                spread = max(spread, (max(traces) - min(traces)) / max(traces))
                for i in range(len(ops)):
                    for j in range(i + 1, len(ops)):
                        den = math.sqrt(norms[i] * norms[j])
                        if den > 0:
                            c = float(np.real(np.sum(ops[i] * np.conj(ops[j])))) / den
                            overlap_min = min(overlap_min, c)
        if overlap_min < 0.999 or spread > 1e-3:
            self.ensemble_model_issue(
                src,
                {"similar": False, "min_overlap": overlap_min},
                f"{a.name} / {b.name}",
            )

    def herald_member_efficiency(self, src: _Source, det_id: str) -> np.ndarray:
        weights = src.emission.source.member_weights
        eta = np.zeros(len(weights))
        n_i = len(src.n_i)
        for k, amp in enumerate(src.members):
            if weights[k] <= 0:
                continue
            rs, ri = self.member_rho(amp, src.idx, n_i)
            total = 0.0
            for role, rho in (("signal", rs), ("idler", ri)):
                h = src.h[det_id][role]
                total += _quadratic_trace(h, rho)
            eta[k] = min(total / weights[k], 1.0)
        return eta

    def pulsed_moments(
        self, src: _Source, a: _Channel, b: _Channel, herald: _Channel | None = None
    ) -> dict[str, Any]:
        # Model: every pulse is entirely in one member k of the mixture (probability w_k;
        # amplitudes carry sqrt(w_k)). Per member the unheralded second-order moment is the
        # thermal one built from the filtered one-photon operators N = Y Y^dagger of the arm
        # (Y = filtered amplitude matrix, rows: arm frequencies and polarizations, columns: the
        # partner photon). The per-pulse average is sum_k w_k Tr(Nh_a,k Nh_b,k) plus the
        # covariance over members of the detected intensities (zero when all members have the
        # same reduced state, as for polarization mixing of a common JSA). The Schmidt number is
        # that of the averaged one-body operator sum_k N_k, i.e. of the reduced density matrix of
        # the mixture. With a herald, the same-pulse triple moment H-a-b is evaluated from the
        # herald-weighted cycle term Re Tr(N_a N_b E_H), so that the bunching factor follows the
        # herald's spectral filter member by member.
        key = (src.inst, a.detector, b.detector, None if herald is None else herald.detector)
        cached = self._moments.get(key)
        if cached is not None:
            return cached
        weights = src.emission.source.member_weights
        k_count = len(weights)
        keys = ("tr_a", "tr_b", "thermal", "h_a", "h_b", "cycle")
        member = {name: np.zeros(k_count) for name in keys}
        by_role: dict[str, dict[str, float]] = {}
        overlap_min, spread, checked = 1.0, 0.0, False
        for role in ROLES:
            stats = self.role_gram_stats(src, role, a, b, herald, weights)
            for name in keys:
                member[name] += stats[name]
            tr_a, tr_b = float(np.sum(stats["tr_a"])), float(np.sum(stats["tr_b"]))
            by_role[role] = {
                "trace_a": tr_a,
                "trace_b": tr_b,
                "schmidt_a": tr_a**2 / stats["tr2_a"] if tr_a > 0 else math.nan,
                "schmidt_b": tr_b**2 / stats["tr2_b"] if tr_b > 0 else math.nan,
            }
            if stats["overlap"] is not None:
                checked = True
                overlap_min = min(overlap_min, stats["overlap"])
                spread = max(spread, stats["spread"])
        active = weights > 0
        w = np.where(active, weights, 1.0)
        t_a = member["tr_a"] / w
        t_b = member["tr_b"] / w
        mean_a, mean_b = float(np.sum(weights * t_a)), float(np.sum(weights * t_b))
        covariance = float(np.sum(weights * t_a * t_b) - mean_a * mean_b)
        thermal = float(np.sum(member["thermal"][active] / weights[active]))
        result: dict[str, Any] = {
            "excess": thermal + covariance,
            "by_role": by_role,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "similar": None if not checked else bool(overlap_min >= 0.999 and spread <= 1e-3),
            "min_overlap": overlap_min,
        }
        if herald is not None:
            triple = member["h_a"] * member["tr_b"] + member["h_b"] * member["tr_a"]
            triple = (triple + 2.0 * member["cycle"])[active] / weights[active]
            eta = self.herald_member_efficiency(src, herald.detector)[active]
            baseline = float(
                np.sum(member["h_a"]) * np.sum(member["tr_b"])
                + np.sum(member["h_b"]) * np.sum(member["tr_a"])
            )
            total = float(np.sum(triple))
            saturated = float(np.sum((1.0 - eta / 2.0) * triple))
            result["bunching_conditional"] = total / baseline - 1.0 if baseline > 0 else 0.0
            result["pair_pair_factor"] = saturated / baseline if baseline > 0 else 1.0
            result["herald_saturation"] = saturated / total if total > 0 else 1.0
        self._moments[key] = result
        return result

    def role_gram_stats(
        self,
        src: _Source,
        role: str,
        a: _Channel,
        b: _Channel,
        herald: _Channel | None,
        weights: np.ndarray,
    ) -> dict[str, Any]:
        # Per member k with x the (arm rows) x (partner columns) amplitude matrix, Y_a = h_a x,
        # Y_b = h_b x, and M the partner-space operator h_H^T conj(h_H) of the herald:
        #   tr_a = |Y_a|^2, thermal = Tr(Y_a^+ Y_a Y_b^+ Y_b) = |Y_a Y_b^+|^2,
        #   h_a = Tr(M Y_a^+ Y_a) (herald-a pair weight), cycle = Re Tr(Y_a^+ Y_a Y_b^+ Y_b M).
        # Arm space (dimension rows): G = x x^+, G_H = x M x^+ and block products h G h^+, one
        # big product per matrix. Partner space (dimension cols): g = x^+ E x with E = h^+ h.
        members = src.members
        k_count = len(members)
        names = ("tr_a", "tr_b", "thermal", "h_a", "h_b", "cycle")
        zeros = {n: np.zeros(k_count) for n in names}
        zeros |= {"tr2_a": 0.0, "tr2_b": 0.0, "overlap": None, "spread": 0.0}
        h_a, h_b = src.h[a.detector][role], src.h[b.detector][role]
        if not (np.any(h_a) or np.any(h_b)):
            return zeros
        partner = "idler" if role == "signal" else "signal"
        h_h = src.h[herald.detector][partner] if herald is not None else None
        n_r, n_p = len(src.grid[role]), len(src.grid[partner])
        rows, cols = 2 * n_r, 2 * n_p
        affordable = k_count <= MAX_SIMILARITY_MEMBERS
        keep = affordable and k_count > 1
        space = GRAM_SPACE or ("col" if cols < rows and affordable else "row")
        dim = cols if space == "col" else rows
        needed = 16.0 * (
            (6 + (2 * k_count if (keep or space == "col") else 0)) * dim * dim
            + (5 + (k_count if space == "col" else 0)) * rows * cols
        )
        check_memory(
            needed,
            self.max_bytes,
            f"the second-order moments of pair source '{src.inst}' "
            f"({rows} x {cols} filtered amplitude matrix, evaluated in the "
            f"{'partner' if space == 'col' else 'arm'} space of dimension {dim})",
            "use a coarser frequency lattice or fewer samples in the pair source",
        )
        m_ops = None
        if h_h is not None:
            m_ops = np.einsum("fck,fcq->fkq", h_h, np.conj(h_h))
        e_a = np.einsum("fsp,fsq->fpq", np.conj(h_a), h_a)
        e_b = np.einsum("fsp,fsq->fpq", np.conj(h_b), h_b)
        out = {n: np.zeros(k_count) for n in names}
        total = {"a": None, "b": None}
        stored: dict[str, list] = {"a": [], "b": []}
        for k, amp in enumerate(members):
            self.cancel_point()
            if weights[k] <= 0.0:
                stored["a"].append(None)
                stored["b"].append(None)
                continue
            x = self.partner_matrix(src, role, amp).reshape(rows, cols)
            if space == "row":
                self.member_arm_space(x, rows, cols, h_a, h_b, m_ops, k, out, total, stored, keep)
            else:
                self.member_partner_space(x, rows, cols, n_p, e_a, e_b, m_ops, k, out, stored, keep)
        active = [k for k in range(k_count) if weights[k] > 0]
        result: dict[str, Any] = dict(out)
        result["overlap"], result["spread"] = None, 0.0
        for label in ("a", "b"):
            tr2, gram = self.mixture_purity(
                space, stored[label], total[label], active, keep, e_a if label == "a" else e_b
            )
            result[f"tr2_{label}"] = tr2
            if gram is not None and label == "a" and len(active) > 1:
                diag = np.sqrt(np.clip(np.diag(gram), 0.0, None))
                den = np.outer(diag, diag)
                sel = np.ix_(active, active)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cos = np.where(den[sel] > 0, gram[sel] / den[sel], 1.0)
                result["overlap"] = float(np.min(cos))
                traces = out["tr_a"][active] / weights[active]
                top = float(np.max(traces))
                result["spread"] = float((top - np.min(traces)) / top) if top > 0 else 0.0
        return result

    @staticmethod
    def block_product(left: np.ndarray, big: np.ndarray, right: np.ndarray) -> np.ndarray:
        # left_f big_{f,g} right_g^+ for per-frequency 2x2 blocks; big has index (f, pol)
        n = left.shape[0]
        g4 = big.reshape(n, 2, right.shape[0], 2)
        mid = np.einsum("fps,fsgt->fpgt", left, g4)
        return np.einsum("fpgt,gqt->fpgq", mid, np.conj(right)).reshape(2 * n, -1)

    def member_arm_space(self, x, rows, cols, h_a, h_b, m_ops, k, out, total, stored, keep):
        n = rows // 2
        g = x @ x.conj().T
        g_diag = np.einsum("fsft->fst", g.reshape(n, 2, n, 2))
        out["tr_a"][k] = float(np.einsum("fps,fst,fpt->", h_a, g_diag, np.conj(h_a)).real)
        out["tr_b"][k] = float(np.einsum("fps,fst,fpt->", h_b, g_diag, np.conj(h_b)).real)
        cross = self.block_product(h_a, g, h_b)
        out["thermal"][k] = float(np.sum(np.abs(cross) ** 2))
        if m_ops is not None:
            x_m = np.einsum("rkf,fkq->rqf", x.reshape(rows, 2, cols // 2), m_ops).reshape(
                rows, cols
            )
            gh = x_m @ x.conj().T
            gh_diag = np.einsum("fsft->fst", gh.reshape(n, 2, n, 2))
            out["h_a"][k] = float(np.einsum("fps,fst,fpt->", h_a, gh_diag, np.conj(h_a)).real)
            out["h_b"][k] = float(np.einsum("fps,fst,fpt->", h_b, gh_diag, np.conj(h_b)).real)
            cba = self.block_product(h_b, gh, h_a)
            out["cycle"][k] = float(np.real(np.sum(cba * cross.T)))
            del gh, x_m, cba
        for label, h in (("a", h_a), ("b", h_b)):
            nk = self.block_product(h, g, h)
            total[label] = nk if total[label] is None else total[label] + nk
            stored[label].append(nk if keep else None)

    def member_partner_space(self, x, rows, cols, n_p, e_a, e_b, m_ops, k, out, stored, keep):
        n = rows // 2
        x3 = x.reshape(n, 2, cols)
        xh = x.conj().T
        xa = np.einsum("fps,fsc->fpc", e_a, x3).reshape(rows, cols)
        xb = np.einsum("fps,fsc->fpc", e_b, x3).reshape(rows, cols)
        ga, gb = xh @ xa, xh @ xb
        out["tr_a"][k] = float(np.real(np.trace(ga)))
        out["tr_b"][k] = float(np.real(np.trace(gb)))
        out["thermal"][k] = float(np.real(np.sum(ga * gb.T)))
        if m_ops is not None:
            ga_d = np.einsum("kfqf->fkq", ga.reshape(2, n_p, 2, n_p))
            gb_d = np.einsum("kfqf->fkq", gb.reshape(2, n_p, 2, n_p))
            out["h_a"][k] = float(np.real(np.einsum("fkq,fqk->", ga_d, m_ops)))
            out["h_b"][k] = float(np.real(np.einsum("fkq,fqk->", gb_d, m_ops)))
            blocks = np.einsum("qfc,ckf->fqk", ga.reshape(2, n_p, cols), gb.reshape(cols, 2, n_p))
            out["cycle"][k] = float(np.real(np.einsum("fqk,fkq->", blocks, m_ops)))
        stored["a"].append(
            (xh if keep else None, xa if keep else None, float(np.sum(np.abs(ga) ** 2)))
        )
        stored["b"].append(
            (xh if keep else None, xb if keep else None, float(np.sum(np.abs(gb) ** 2)))
        )

    @staticmethod
    def mixture_purity(space: str, stored: list, total, active: list, keep: bool, e_op):
        # Tr(N^2) of the averaged arm operator N = sum_k N_k, and (when affordable) the matrix
        # Tr(N_k N_l) of member overlaps.
        if not active:
            return 0.0, None
        if space == "row":
            tr2 = float(np.real(np.sum(total * total.T)))
            if not keep:
                return tr2, None
            gram = np.zeros((len(stored), len(stored)))
            for i in active:
                for j in active:
                    gram[i, j] = float(np.real(np.sum(stored[i] * stored[j].T)))
            return tr2, gram
        # partner space: Tr(N_k N_l) = |x_k^+ E x_l|^2 (stored: x_k^+ and E x_k)
        gram = np.zeros((len(stored), len(stored)))
        for i in active:
            gram[i, i] = stored[i][2]
        if len(active) > 1:
            for i in active:
                for j in active:
                    if j < i:
                        gram[i, j] = gram[j, i]
                    elif j > i:
                        gram[i, j] = float(np.sum(np.abs(stored[i][0] @ stored[j][1]) ** 2))
        return float(np.sum(gram)), gram

    def partner_matrix(self, src: _Source, role: str, amp: np.ndarray) -> np.ndarray:
        idx = src.idx
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


MAX_PAIR_CHANNELS = 24


def simulate_rates(
    setup: Setup,
    library: Library,
    classical: SimulationResult | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
    cache: RunCache | None = None,
    max_memory_bytes: float | None = None,
) -> RateResult:
    """Run the photon-rate engine.

    ``should_cancel`` is polled between stages and inside the long loops; when it returns True
    the run stops with ``Cancelled``. ``cache`` (a ``RunCache``) lets parsed parts and the
    ``emit_pairs`` results of components that declare ``emit_cache_key`` be reused across runs.
    ``max_memory_bytes`` bounds the arrays the engine is allowed to allocate for one stage.
    """
    engine = _Engine(setup, library, classical, should_cancel, cache, max_memory_bytes)
    engine.discover()
    engine.emit_sources()
    engine.transfer_sources()
    engine.cancel_point()
    engine.background()
    engine.source_diagnostics()
    analyses: dict[str, dict[str, Any]] = {}
    for spec in setup.analyses:
        engine.cancel_point()
        try:
            name, result = engine.analyse(spec)
        except AnalysisSpecError as exc:
            engine.issue("error", "bad_analysis", "analyses", str(exc))
            continue
        except ResourceLimitError as exc:
            subject = f"analysis '{spec.get('name', spec.get('type', 'analysis'))}'"
            engine.issue("error", "memory_limit", subject, f"{exc}; the analysis was skipped")
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
    if len(names) <= MAX_PAIR_CHANNELS and engine.sources:
        try:
            for i, x in enumerate(names):
                for y in names[i + 1 :]:
                    engine.cancel_point()
                    rate = engine.pair_rate(engine.channels[x], engine.channels[y])
                    pairs[(x, y)] = {"coincidence_rate_hz": rate}
        except ResourceLimitError as exc:
            engine.issue(
                "error", "memory_limit", "setup", f"{exc}; the coincidence table was skipped"
            )
            pairs = {}
    elif engine.sources:
        engine.issue(
            "warning",
            "pair_table_truncated",
            "setup",
            f"{len(names)} tagger channels are connected; the all-pairs coincidence table is "
            f"only built for up to {MAX_PAIR_CHANNELS} channels (use analyses for the others)",
        )
    sources = {
        s.inst: {
            "rate_hz": s.rate,
            "dnu_hz": s.dnu,
            "pulsed": s.pulsed,
            "repetition_rate_hz": s.emission.source.repetition_rate,
            "signal_wavelength_m": C_LIGHT / s.nu0["signal"] if s.nu0["signal"] > 0 else None,
            "idler_wavelength_m": C_LIGHT / s.nu0["idler"] if s.nu0["idler"] > 0 else None,
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
    links = {}
    for link_id, entry in engine.link_photons.items():
        total = sum(sum(roles.values()) for roles in entry["by_source"].values())
        links[link_id] = {"photons_hz": total, **entry}
    issues = list(engine.classical.issues)
    issues += [i for i in engine.issues if i not in issues]
    return RateResult(
        engine.classical, channels, analyses, issues, engine.det_data, sources, pairs, links
    )
