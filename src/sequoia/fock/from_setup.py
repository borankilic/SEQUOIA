import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import constants, sparse

from sequoia.components.base import Component, Context, PortDirection, PortKind, get_component
from sequoia.data import Scalar
from sequoia.engines.cache import RunCache, check_cancel
from sequoia.engines.classical import LinkState, build_network, propagate_network
from sequoia.fock.analysis import ExpectedCounts, Samples, expected_counts, sample
from sequoia.fock.circuit import (
    BeamSplitter,
    Block,
    Circuit,
    FockError,
    Loss,
    PhaseShifter,
    bs_matrix,
)
from sequoia.fock.compile import (
    CompiledNetwork,
    FockCompileError,
    _guess_wavelength,
    _psd_sqrt,
    _reachable,
    compile_setup_network,
)
from sequoia.fock.engine import Limits
from sequoia.fock.experiment import DetectorSpec, Experiment, FockResult, run_experiment
from sequoia.fock.inputs import (
    InputMixture,
    PureState,
    Term,
    pair_number_mixture,
    pair_number_pmf,
    product_mixture,
)
from sequoia.library.store import Library
from sequoia.light.gaussian import GaussianMode, aperture_transmission
from sequoia.setup.model import Setup

TAIL = 1e-4
MERGE_RTOL = 1e-3
PRIVATE_OFFSET = 5000
SOURCE_TAG_STRIDE = 10_000
MAX_N = 12
KIND_TO_MODEL = {"SPAD": "threshold", "SNSPD": "threshold"}
WINDOW_RTOL = 1e-9
_TOL = 1e-9


class IncompatibleWindowsError(FockError):
    pass


@dataclass(frozen=True)
class ComponentRow:
    id: str
    type: str
    category: str
    part: str
    summary: str
    ports: str


@dataclass(frozen=True)
class InputDescription:
    source: str
    kind: str
    text: str
    window_s: float
    photons: str = ""
    mean_pairs: float | None = None
    schmidt: float | None = None
    n_max: int | None = None
    omitted: float = 0.0
    wavelengths_nm: tuple[float, ...] = ()
    used: bool = True


@dataclass
class _Chan:
    inst: str
    port: str
    wavelength: float
    mode: GaussianMode | None
    basis: tuple[tuple[str, str], np.ndarray] | None
    species: str


@dataclass
class _Fock:
    inst: str
    photons: int
    tag: int
    jones: np.ndarray
    wavelength: float
    rep: float
    mode: GaussianMode


@dataclass
class _Pair:
    inst: str
    signal_port: str
    idler_port: str
    pulsed: bool
    rate: float
    rep: float | None
    dnu: float
    mu: float
    window: float
    schmidt: float | None
    rho_pol: np.ndarray
    eta: float
    wl_signal: float
    wl_idler: float
    mode_signal: GaussianMode
    mode_idler: GaussianMode
    notes: list[str] = field(default_factory=list)


@dataclass
class _Found:
    net: object
    fock: list[_Fock]
    pairs: list[_Pair]
    issues: list = field(default_factory=list)


@dataclass
class _Plan:
    channels: list[_Chan]
    keys: list[tuple[str, str]]
    make: Callable[[Callable[[int, int], int]], InputMixture]
    descriptions: list[InputDescription]
    notes: list[str]
    window: float
    omitted: float
    fock_angles: dict[str, float]


DILUTE_LOAD = 0.1


@dataclass
class SetupExperiment:
    experiment: Experiment
    compiled: CompiledNetwork
    input_labels: list[str]
    detector_ids: list[str]
    output_labels: list[str]
    inputs: list[InputDescription]
    notes: list[str]
    components: list[ComponentRow]
    window_s: float
    rate_hz: float
    duration_s: float
    omitted_weight: float
    plan: _Plan
    setup: Setup
    library: Library
    dead_times: dict[str, float] = field(default_factory=dict)

    @property
    def n_detectors(self) -> int:
        return len(self.detector_ids)


@dataclass
class SetupResult:
    setup_experiment: SetupExperiment
    result: FockResult
    detected: dict[tuple[int, ...], float]
    detector_ids: list[str]
    photon_numbers: dict[tuple[int, ...], float]
    per_mode: dict[tuple[int, ...], float]
    counts: ExpectedCounts
    samples: Samples | None
    omitted_weight: float
    messages: list[str]
    dead_time_load: dict[str, float] = field(default_factory=dict)

    @property
    def mass(self) -> float:
        return float(sum(self.detected.values()))

    @property
    def outside_dilute_regime(self) -> dict[str, float]:
        return {d: v for d, v in self.dead_time_load.items() if v > DILUTE_LOAD}


def _fmt_value(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, Scalar):
        return f"{v.value:g}" if v.unit.strip() in ("", "1") else f"{v.value:g} {v.unit}"
    if isinstance(v, (bool, int, str)):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    if type(v).__name__ == "Curve":
        return "curve"
    return None


def list_components(setup: Setup, library: Library) -> list[ComponentRow]:
    net = build_network(setup, library)
    rows: list[ComponentRow] = []
    for inst in setup.instances:
        comp = net.components[inst.id]
        parts = []
        try:
            for k, v in comp.settings.model_dump(mode="json").items():
                if v is not None:
                    parts.append(f"{k}={v}")
        except Exception:
            parts = []
        shown = 0
        for name in type(comp.params).model_fields:
            text = _fmt_value(getattr(comp.params, name, None))
            if text is None or name in ("name", "description"):
                continue
            parts.append(f"{name}={text}")
            shown += 1
            if shown >= 5:
                break
        ports = []
        for p in comp.ports():
            tag = "" if p.kind == PortKind.ELECTRICAL else ":H,V"
            ports.append(f"{p.name}{tag} ({'in' if p.direction == PortDirection.IN else 'out'})")
        rows.append(
            ComponentRow(
                inst.id,
                comp.type_name,
                comp.category,
                inst.part,
                ", ".join(parts),
                ", ".join(ports),
            )
        )
    for link in setup.links:
        if link.kind == "cable":
            text = f"{link.src} -> {link.dst}, delay {_fmt_value(link.delay)}"
            part = ""
        else:
            length = net.lengths.get(link.id)
            text = f"{link.src} -> {link.dst}"
            if length is not None:
                text += f", length {length * 1e3:.6g} mm"
            part = getattr(link, "fiber", "") or ""
            if link.kind == "fiber" and link.connector_loss_db:
                text += f", connector loss {link.connector_loss_db:g} dB"
        rows.append(ComponentRow(link.id, f"link ({link.kind})", "link", part, text, ""))
    return rows


def _pair_data(src, mu_window: float) -> tuple[float | None, np.ndarray, float, list[str]]:
    notes: list[str] = []
    members = np.asarray(src.members)
    total = float(np.sum(np.abs(members) ** 2))
    flat = members.transpose(1, 2, 0, 3, 4).reshape(4, -1)
    rho = (flat @ flat.conj().T) / total
    w_s = np.real(np.trace(src.rho["signal"], axis1=1, axis2=2))
    w_i = np.real(np.trace(src.rho["idler"], axis1=1, axis2=2))
    common, ps_, pi_ = np.intersect1d(src.grid["signal"], src.grid["idler"], return_indices=True)
    if len(common) == 0 or w_s.sum() <= 0 or w_i.sum() <= 0:
        eta = 0.0
    else:
        eta = float((np.sum(np.sqrt(w_s[ps_] * w_i[pi_])) / math.sqrt(w_s.sum() * w_i.sum())) ** 2)
    k_idx, a_idx, b_idx, j_idx, m_idx = np.nonzero(members)
    vals = members[k_idx, a_idx, b_idx, j_idx, m_idx]
    l_idx = src.idx[j_idx, m_idx]
    n_s, n_i = len(src.grid["signal"]), len(src.n_i)
    pol = a_idx * 2 + b_idx + 4 * k_idx
    n_members = len(members)
    if n_s <= n_i:
        row, col, shape = j_idx, pol.astype(np.int64) * n_i + l_idx, (n_s, n_members * 4 * n_i)
    else:
        row, col, shape = l_idx, pol.astype(np.int64) * n_s + j_idx, (n_i, n_members * 4 * n_s)
    y = sparse.coo_matrix((vals, (row, col)), shape=shape).tocsr()
    gram = y @ y.conj().T
    tr2 = float(np.sum(np.abs(gram.data) ** 2))
    tr1 = float(np.sum(np.abs(vals) ** 2))
    schmidt = max(tr1**2 / tr2, 1.0) if tr2 > 0 else None
    if schmidt is None:
        notes.append(f"{src.inst}: Schmidt number unavailable, K = 1 assumed")
    return schmidt, rho, eta, notes


def _discover(
    setup: Setup,
    library: Library,
    classical,
    should_cancel,
    cache: RunCache | None,
) -> _Found:
    net = build_network(setup, library)
    fock: list[_Fock] = []
    from sequoia.components.sources import _launch_mode

    for inst_id, comp in net.components.items():
        occ = getattr(comp, "fock_occupation", None)
        if occ is None:
            continue
        n, tag, jones, wl, rep = occ()
        fock.append(
            _Fock(
                inst_id,
                int(n),
                int(tag),
                np.asarray(jones, dtype=complex),
                float(wl),
                float(rep),
                _launch_mode(comp.params, float(wl), "a Fock source"),
            )
        )
    pairs: list[_Pair] = []
    issues: list = []
    has_pair = any(type(c).emit_pairs is not Component.emit_pairs for c in net.components.values())
    if has_pair:
        from sequoia.engines.rates import _Engine

        engine = _Engine(setup, library, classical, should_cancel, cache)
        engine.emit_sources()
        net = engine.net
        issues = list(engine.issues)
        for src in engine.sources:
            check_cancel(should_cancel)
            ps = src.emission.source
            window = 1.0 / ps.repetition_rate if src.pulsed else 1.0 / ps.dnu
            schmidt, rho, eta, notes = _pair_data(src, window)
            pairs.append(
                _Pair(
                    src.inst,
                    src.emission.signal_port,
                    src.emission.idler_port,
                    src.pulsed,
                    float(src.rate),
                    ps.repetition_rate,
                    float(ps.dnu),
                    float(src.rate) * window,
                    window,
                    schmidt,
                    rho,
                    eta,
                    constants.c / src.nu0["signal"],
                    constants.c / src.nu0["idler"],
                    ps.signal_mode,
                    ps.idler_mode,
                    notes,
                )
            )
    return _Found(net, fock, pairs, issues)


def _angle_deg(jones) -> float:
    return math.degrees(math.atan2(float(np.real(jones[1])), float(np.real(jones[0]))))


def _tail(mu: float, k: float, n: int) -> float:
    return max(0.0, 1.0 - sum(pair_number_pmf(i, mu, k) for i in range(n + 1)))


def _choose_n_max(
    pairs: list[_Pair], fixed: int, limits: Limits, override: int | None
) -> list[int]:
    if override is not None:
        return [max(0, int(override))] * len(pairs)
    need = []
    for p in pairs:
        k = p.schmidt or 1.0
        n = 0
        while n < MAX_N and _tail(p.mu, k, n) > TAIL:
            n += 1
        need.append(n)
    allowed = max(0, (limits.max_photons - fixed) // 2)
    while sum(need) > allowed and any(need):
        need[need.index(max(need))] -= 1
    return need


def _pol_state_branches(rho: np.ndarray) -> list[tuple[float, np.ndarray]]:
    w, v = np.linalg.eigh((rho + rho.conj().T) / 2.0)
    out = [(float(x), v[:, i]) for i, x in enumerate(w) if x > 1e-9]
    total = sum(x for x, _ in out)
    return [(x / total, c) for x, c in out]


def _make_plan(
    found: _Found,
    wavelength: float | None,
    n_max: int | None,
    limits: Limits,
    sources: Sequence[str] | None,
) -> _Plan:
    notes: list[str] = []
    fock = [f for f in found.fock if sources is None or f.inst in sources]
    pairs = [p for p in found.pairs if sources is None or p.inst in sources]
    fock = [f for f in fock if f.photons > 0] if pairs or len(fock) > 1 else fock
    if not fock and not pairs:
        raise FockCompileError(
            "no photon-number source in the setup: add a FockSource or an SPDC crystal / "
            "entangled-pair source (a laser is a classical beam and has no photon-number input)"
        )
    windows = [(f.inst, 1.0 / f.rep) for f in fock] + [(p.inst, p.window) for p in pairs]
    lo = min(w for _, w in windows)
    if any(w > lo * (1.0 + WINDOW_RTOL) for _, w in windows):
        listing = ", ".join(f"'{i}': {w:.6g} s" for i, w in windows)
        raise IncompatibleWindowsError(
            f"the photon sources have different time windows ({listing}); per-window statistics "
            "of independent sources cannot be combined. Give them the same repetition rate or "
            "bandwidth, or select the sources to use"
        )
    window = windows[0][1]
    channels: list[_Chan] = []
    keys: list[tuple[str, str]] = []

    def channel(inst, port, wl, mode, basis, species) -> int:
        for i, c in enumerate(channels):
            if c.inst == inst and c.port == port and c.basis is None and basis is None:
                if abs(c.wavelength - wl) > MERGE_RTOL * wl:
                    notes.append(
                        f"{inst}.{port}: signal and idler leave one port at different "
                        f"wavelengths ({c.wavelength * 1e9:.2f} and {wl * 1e9:.2f} nm); path "
                        "phases use their mean, their distinguishability enters through the "
                        "spectral overlap"
                    )
                if species not in c.species:
                    c.species = f"{c.species}+{species}"
                c.wavelength = 0.5 * (c.wavelength + wl)
                return i
        channels.append(_Chan(inst, port, wl, mode, basis, species))
        keys.append((inst, port))
        return len(channels) - 1

    fock_chan: list[int] = []
    fock_angles: dict[str, float] = {}
    for f in fock:
        theta = math.atan2(float(np.real(f.jones[1])), float(np.real(f.jones[0])))
        c, s = math.cos(theta), math.sin(theta)
        names = ("pol", "perp")
        fock_chan.append(
            channel(
                f.inst,
                "out",
                f.wavelength,
                f.mode,
                (names, np.array([[c, -s], [s, c]], dtype=complex)),
                "photons",
            )
        )
        if abs(theta) > 1e-12:
            fock_angles[f.inst] = theta
    pair_chan: list[tuple[int, int]] = []
    for p in pairs:
        ci = channel(p.inst, p.signal_port, p.wl_signal, p.mode_signal, None, "signal")
        cj = channel(p.inst, p.idler_port, p.wl_idler, p.mode_idler, None, "idler")
        pair_chan.append((ci, cj))
    if wavelength is not None:
        for c in channels:
            c.wavelength = float(wavelength)
        notes.append(f"all photons use the given wavelength {wavelength * 1e9:.2f} nm")
    fixed = sum(f.photons for f in fock)
    n_pair = _choose_n_max(pairs, fixed, limits, n_max)
    descriptions: list[InputDescription] = []
    omitted_total = 0.0
    keep = 1.0
    for f in found.fock:
        used = f in fock
        descriptions.append(
            InputDescription(
                f.inst,
                "fock",
                f"FockSource {f.inst}: {f.photons} photon(s) per pulse, tag {f.tag}, polarization "
                f"{_angle_deg(f.jones):.4g} deg, "
                f"{f.wavelength * 1e9:.2f} nm, {f.rep:.6g} Hz" + ("" if used else " (not used)"),
                1.0 / f.rep,
                photons=str(f.photons),
                wavelengths_nm=(f.wavelength * 1e9,),
                used=used,
            )
        )
    for p, nm in zip(pairs, n_pair, strict=True):
        k = p.schmidt or 1.0
        tail = _tail(p.mu, k, nm)
        keep *= 1.0 - tail
        omitted_total = 1.0 - keep
        notes.extend(p.notes)
        if p.schmidt is None:
            notes.append(f"{p.inst}: Schmidt number unavailable, K = 1 assumed")
        if tail > TAIL:
            notes.append(
                f"{p.inst}: pair numbers above {nm} (weight {tail:.3g}) are not simulated; "
                "raise the photon limit or lower the pump power"
            )
        pol_rank = len(_pol_state_branches(p.rho_pol))
        descriptions.append(
            InputDescription(
                p.inst,
                "pair",
                f"pair source {p.inst}: mu = {p.mu:.4g} pairs/window ({p.rate:.4g} pairs/s x "
                f"{p.window:.4g} s), K = {k:.4g} Schmidt modes, signal {p.wl_signal * 1e9:.2f} nm, "
                f"idler {p.wl_idler * 1e9:.2f} nm, signal-idler overlap {p.eta:.3f}, "
                f"polarization state rank {pol_rank}, pair numbers 0..{nm} (omitted {tail:.2g})",
                p.window,
                photons=f"0..{2 * nm}",
                mean_pairs=p.mu,
                schmidt=k,
                n_max=nm,
                omitted=tail,
                wavelengths_nm=(p.wl_signal * 1e9, p.wl_idler * 1e9),
            )
        )
    if any(p.eta < 0.999 for p in pairs):
        notes.append(
            "signal-idler distinguishability: each idler photon interferes with the signal "
            "photons with probability eta (squared overlap of the signal and idler spectral "
            "intensity amplitudes, phases neglected), otherwise it is distinguishable"
        )

    def make(mode_of: Callable[[int, int], int]) -> InputMixture:
        parts: list[InputMixture | PureState] = []
        for f, ci in zip(fock, fock_chan, strict=True):
            m = mode_of(ci, 0)
            parts.append(PureState((Term(1.0, tuple((m, f.tag) for _ in range(f.photons))),)))
        for idx, (p, (ci, cj), nm) in enumerate(zip(pairs, pair_chan, n_pair, strict=True)):
            base = SOURCE_TAG_STRIDE * (idx + 1)
            branches = _pol_state_branches(p.rho_pol)

            def builder(j, p=p, ci=ci, cj=cj, base=base, branches=branches):
                tag_s = base + j
                tags_i = []
                if p.eta > 0.999:
                    tags_i = [(tag_s, 1.0)]
                elif p.eta < 1e-3:
                    tags_i = [(base + PRIVATE_OFFSET + j, 1.0)]
                else:
                    tags_i = [(tag_s, p.eta), (base + PRIVATE_OFFSET + j, 1.0 - p.eta)]
                out = []
                for w, vec in branches:
                    for tag_i, wi in tags_i:
                        acc: dict = {}
                        for ps in range(2):
                            for pi in range(2):
                                c = vec[ps * 2 + pi]
                                if abs(c) < 1e-12:
                                    continue
                                key = tuple(
                                    sorted(((mode_of(ci, ps), tag_s), (mode_of(cj, pi), tag_i)))
                                )
                                fact = math.sqrt(
                                    math.prod(math.factorial(x) for x in Counter(key).values())
                                )
                                acc[key] = acc.get(key, 0j) + c * fact
                        state = PureState(tuple(Term(a, k_) for k_, a in acc.items())).normalised()
                        out.append((w * wi, state))
                return InputMixture(tuple(out))

            parts.append(pair_number_mixture(builder, p.mu, p.schmidt or 1.0, nm))
        return product_mixture(parts)

    return _Plan(channels, keys, make, descriptions, notes, window, omitted_total, fock_angles)


def build_experiment_from_setup(
    setup: Setup,
    library: Library,
    *,
    wavelength: float | None = None,
    n_max: int | None = None,
    duration: float = 1.0,
    detectors_mode: str = "setup",
    should_cancel: Callable[[], bool] | None = None,
    cache: RunCache | None = None,
    classical=None,
    limits: Limits | None = None,
    sources: Sequence[str] | None = None,
) -> SetupExperiment:
    """Photon-number experiment for the circuit drawn in ``setup``.

    Inputs are found automatically: every FockSource (N photons per pulse) and every pair source
    (thermal pair-number statistics with mean ``rate * window`` and K Schmidt modes). All
    sources must share one detection window. Detectors are the setup's single-photon detectors
    reachable from the sources: ``detectors_mode`` "setup" gives threshold detectors for SPAD and
    SNSPD, "pnr" ideal photon-number-resolving ones.
    """
    if detectors_mode not in ("setup", "pnr"):
        raise FockError("detectors_mode must be 'setup' or 'pnr'")
    limits = limits or Limits()
    found = _discover(setup, library, classical, should_cancel, cache)
    plan = _make_plan(found, wavelength, n_max, limits, sources)
    net = found.net
    check_cancel(should_cancel)
    specs = [f"{c.inst}.{c.port}" for c in plan.channels]
    reach = _reachable(net, {c.inst for c in plan.channels})
    det_ids = [
        i for i, c in net.components.items() if c.detector_model() is not None and i in reach
    ]
    if not det_ids:
        raise FockCompileError(
            "no single-photon detector (SPAD, SNSPD) is reachable from the photon sources; "
            "a photodiode reads classical power and has no photon-number output"
        )
    kinds = {
        d: (
            "pnr"
            if detectors_mode == "pnr"
            else KIND_TO_MODEL.get(net.components[d].detector_model().kind, "pnr")
        )
        for d in det_ids
    }
    compiled = compile_setup_network(
        setup,
        library,
        inputs=specs,
        detectors=det_ids,
        wavelengths=[c.wavelength for c in plan.channels],
        bases=[c.basis for c in plan.channels],
        seed_modes=[c.mode for c in plan.channels],
        species=[c.species for c in plan.channels],
        gate_window=plan.window,
        should_cancel=should_cancel,
        detector_kind=kinds,
        net=net,
    )
    mixture = plan.make(lambda c, p: 2 * c + p)
    n = compiled.circuit.n_modes
    labels = [str(m) for m in compiled.input_modes] + [""] * (n - len(compiled.input_modes))
    window = plan.window
    exp = Experiment(
        compiled.circuit,
        [0] * n,
        detectors=list(compiled.detectors),
        mode_labels=labels,
        name=setup.name,
        limits=limits,
        input_mixture=mixture,
        rate_hz=1.0 / window,
        window_s=window,
        duration_s=float(duration),
    )
    notes = list(plan.notes)
    notes.extend(compiled.warnings)
    notes.extend(f"{i.subject}: {i.message}" for i in found.issues if i.severity == "warning")
    notes.extend(mixture.notes)
    lams = sorted({round(c.wavelength * 1e9, 3) for c in plan.channels})
    notes.append(
        "photon species of "
        + ", ".join(f"{x:g} nm" for x in lams)
        + " accumulate path phase 2 pi OPL / lambda at their own wavelength; a single wavelength "
        "per species (no spectrum), perfect spatial-mode overlap where paths recombine, no "
        "timing, jitter or dead time"
    )
    notes.append(
        "pair-number statistics: thermal with K equal-weight Schmidt modes; photons of one "
        "Schmidt mode bunch, pairs in different modes are mutually distinguishable"
    )
    if plan.omitted > TAIL:
        notes.append(
            f"omitted input weight (pair numbers beyond the truncation): {plan.omitted:.3g}"
        )
    return SetupExperiment(
        exp,
        compiled,
        [str(m) for m in compiled.input_modes],
        det_ids,
        [str(m) for m in compiled.output_modes],
        plan.descriptions,
        list(dict.fromkeys(notes)),
        list_components(setup, library),
        window,
        1.0 / window,
        float(duration),
        plan.omitted,
        plan,
        setup,
        library,
        {d: float(net.components[d].detector_model().dead_time) for d in det_ids},
    )


def list_inputs(
    setup: Setup,
    library: Library,
    *,
    n_max: int | None = None,
    should_cancel=None,
    cache=None,
    classical=None,
    limits: Limits | None = None,
) -> list[InputDescription]:
    found = _discover(setup, library, classical, should_cancel, cache)
    out: list[InputDescription] = []
    try:
        out = _make_plan(found, None, n_max, limits or Limits(), None).descriptions
    except IncompatibleWindowsError:
        pass
    except FockCompileError:
        return []
    if out:
        return out
    rows = []
    for f in found.fock:
        rows.append(
            InputDescription(
                f.inst,
                "fock",
                f"FockSource {f.inst}: {f.photons} photon(s), tag {f.tag}",
                1.0 / f.rep,
                photons=str(f.photons),
            )
        )
    for p in found.pairs:
        rows.append(
            InputDescription(
                p.inst,
                "pair",
                f"pair source {p.inst}: mu = {p.mu:.4g}, K = {p.schmidt or 1.0:.4g}",
                p.window,
                mean_pairs=p.mu,
                schmidt=p.schmidt,
            )
        )
    return rows


def run_setup_experiment(
    se: SetupExperiment,
    *,
    shots: int = 0,
    seed: int | None = None,
    backend: str = "native",
    duration: float | None = None,
    should_cancel=None,
    progress=None,
) -> SetupResult:
    exp = replace(se.experiment, shots=0, backend=backend, conditions={})
    res = run_experiment(exp, should_cancel, progress)
    n_det = se.n_detectors
    n_out = 2 * n_det
    detected: dict[tuple[int, ...], float] = {}
    for s, p in res.probabilities.items():
        detected[s[:n_det]] = detected.get(s[:n_det], 0.0) + p
    detected = dict(sorted(detected.items()))
    per_mode: dict[tuple[int, ...], float] = {}
    for s, p in res.raw.probs.items():
        per_mode[s[:n_out]] = per_mode.get(s[:n_out], 0.0) + p
    per_mode = dict(sorted(per_mode.items()))
    pn: dict[tuple[int, ...], float] = {}
    for s, p in per_mode.items():
        key = tuple(s[2 * k] + s[2 * k + 1] for k in range(n_det))
        pn[key] = pn.get(key, 0.0) + p
    dur = se.duration_s if duration is None else float(duration)
    counts = expected_counts(detected, se.rate_hz, None, dur)
    samples = sample(detected, shots, seed) if shots > 0 and detected else None
    messages = list(res.messages)
    load: dict[str, float] = {}
    for k, d in enumerate(se.detector_ids):
        p_click = sum(p for s, p in detected.items() if s[k] > 0)
        load[d] = p_click * counts.windows_per_second * se.dead_times.get(d, 0.0)
        if load[d] > DILUTE_LOAD:
            messages.append(
                f"'{d}': the potential click rate {p_click * counts.windows_per_second:.4g}/s "
                f"times the dead time {se.dead_times[d] * 1e9:.4g} ns is {load[d]:.3g}, outside "
                f"the dilute regime (< {DILUTE_LOAD:g}); this model omits dead time, so its "
                "click rates are potential clicks, not recorded counts. Attenuate the input or "
                "use the Monte Carlo engine for recorded tag rates"
            )
    return SetupResult(
        se,
        res,
        detected,
        list(se.detector_ids),
        dict(sorted(pn.items())),
        per_mode,
        counts,
        samples,
        res.raw.omitted_weight,
        messages,
        load,
    )


@dataclass
class Extraction:
    circuit: Circuit
    input_modes: dict[str, tuple[int, int]]
    detector_modes: dict[str, tuple[int, int]]
    mode_labels: list[str]
    wavelength: float
    verified: bool
    max_error: float
    fallback: bool
    notes: list[str]
    fock_angles: dict[str, float]


def _scalar_one(c: complex, m: int, label: str, out: list, tol: float = 1e-9) -> None:
    amp = abs(c)
    phase = math.atan2(c.imag, c.real)
    if abs(phase) > tol:
        out.append(PhaseShifter(m, phase, label))
    if amp < 1.0 - tol:
        out.append(Loss(m, min(1.0, max(0.0, 1.0 - amp * amp)), "optical", label))


def _scalar_elements(c: complex, m_h: int, m_v: int, label: str, out: list) -> None:
    _scalar_one(c, m_h, label, out)
    _scalar_one(c, m_v, label, out)


def _two_pol_elements(j: np.ndarray, m_h: int, m_v: int, label: str, out: list) -> None:
    j = np.asarray(j, dtype=complex)
    if np.allclose(j, j[0, 0] * np.eye(2), atol=1e-10):
        if not np.isclose(j[0, 0], 1.0, atol=1e-10):
            _scalar_elements(complex(j[0, 0]), m_h, m_v, label, out)
        return
    out.append(Block((m_h, m_v), j, label=label))


def extract_circuit(
    setup: Setup,
    library: Library,
    wavelength: float | None = None,
    *,
    net=None,
    should_cancel=None,
    verify: bool = True,
    seed_modes: dict[str, GaussianMode] | None = None,
) -> Extraction:
    """Decompose the drawn network into editable circuit elements on a mode register with one
    mode per (link, polarization) path. Every element carries the id of its component or link.
    The transfer matrix is checked against the exact compilation; when they differ, one block
    holding the compiled transfer replaces the decomposition (``fallback``)."""
    net = net if net is not None else build_network(setup, library)
    notes: list[str] = []
    photon_src = [
        i
        for i, c in net.components.items()
        if hasattr(c, "fock_occupation") or type(c).emit_pairs is not Component.emit_pairs
    ]
    if not photon_src:
        photon_src = [i for i, c in net.components.items() if c.category == "sources"]
    starts: list[tuple[str, str]] = []
    for inst in photon_src:
        for p in net.components[inst].ports():
            if p.direction == PortDirection.OUT and f"{inst}.{p.name}" in net.out_link:
                starts.append((inst, p.name))
    if not starts:
        raise FockCompileError("no source output port is connected; nothing to decompose")
    fock = {
        i: net.components[i].fock_occupation()
        for i in photon_src
        if hasattr(net.components[i], "fock_occupation")
    }
    if seed_modes is None and any(
        type(net.components[i]).emit_pairs is not Component.emit_pairs for i in photon_src
    ):
        found = _discover(setup, library, None, should_cancel, None)
        seed_modes = {}
        for p in found.pairs:
            seed_modes[f"{p.inst}.{p.idler_port}"] = p.mode_idler
            seed_modes[f"{p.inst}.{p.signal_port}"] = p.mode_signal
        if wavelength is None and not fock and found.pairs:
            wavelength = found.pairs[0].wl_signal
    if wavelength is not None:
        lam = float(wavelength)
    elif fock:
        lam = float(next(iter(fock.values()))[3])
    else:
        lam = _guess_wavelength(net, notes)
    wl = np.array([lam])
    ctx = Context(issues=[])
    states = {
        link.id: LinkState(link.id, link.kind, link.src, link.dst, None) for link in net.setup.links
    }
    from sequoia.light.state import Beam

    def seed_for(inst: str, port: str) -> Beam:
        comp = net.components[inst]
        mode = GaussianMode.from_waist(lam, 0.5e-3)
        if seed_modes is not None and f"{inst}.{port}" in seed_modes:
            mode = seed_modes[f"{inst}.{port}"]
        if hasattr(comp, "fock_occupation"):
            from sequoia.components.sources import _launch_mode

            mode = _launch_mode(comp.params, lam, "a Fock source")
        return Beam(
            f"fock:{inst}", wl, np.eye(2, dtype=complex)[None].copy(), 0.0, 0.0, mode, 0.0, ()
        )

    def emit(inst_id, comp, ctx_):
        return [(p, seed_for(i, p)) for i, p in starts if i == inst_id]

    arrivals, ends = propagate_network(net, ctx, emit, states=states, should_cancel=should_cancel)
    reach = _reachable(net, {i for i, _ in starts})
    in_link = {link.dst: link for link in net.setup.links if link.kind != "cable"}
    labels: list[str] = []
    elements: list = []

    def new_mode(label: str) -> int:
        labels.append(label)
        return len(labels) - 1

    link_modes: dict[str, tuple[int, int]] = {}
    input_modes: dict[str, tuple[int, int]] = {}
    detector_modes: dict[str, tuple[int, int]] = {}
    start_ports: dict[str, list[str]] = {}
    for i, p in starts:
        start_ports.setdefault(i, []).append(p)
    eye = np.eye(2, dtype=complex)

    def unit(beam: Beam) -> Beam:
        from dataclasses import replace as rep

        return rep(beam, jones=eye[None].copy(), opl=0.0, gpl=0.0, history=())

    def emit_links(inst: str, out_modes: dict[str, tuple[int, int]]) -> None:
        for port, modes in out_modes.items():
            link = net.out_link.get(f"{inst}.{port}")
            if link is None:
                continue
            link_modes[link.id] = modes
            start_beams = states[link.id].start_beams
            if not start_beams:
                continue
            seed = unit(start_beams[0])
            if link.kind == "free_space":
                length = net.lengths[link.id]
                phase = 2.0 * math.pi * length / lam
                j = np.exp(1j * phase) * eye
            else:
                part = net.library.load_part(link.fiber)
                fiber = get_component(part.type)(part.params)
                fctx = Context(issues=[])
                end, _ = fiber.transmit(seed, link, fctx, link.id)
                j = end.jones[0] * np.exp(2j * np.pi * end.opl / lam)
            _two_pol_elements(j, modes[0], modes[1], link.id, elements)

    for inst in net.order:
        if inst not in reach:
            continue
        check_cancel(should_cancel)
        comp = net.components[inst]
        if inst in start_ports:
            out_modes = {}
            for port in start_ports[inst]:
                m = (new_mode(f"{inst}.{port}:H"), new_mode(f"{inst}.{port}:V"))
                input_modes[f"{inst}.{port}"] = m
                out_modes[port] = m
            if inst in fock:
                jones = np.asarray(fock[inst][2], dtype=complex)
                c, s = float(np.real(jones[0])), float(np.real(jones[1]))
                if abs(s) > 1e-12:
                    m = out_modes[start_ports[inst][0]]
                    elements.append(
                        Block(
                            m,
                            np.array([[c, -s], [s, c]], dtype=complex),
                            label=f"{inst} polarization",
                        )
                    )
            emit_links(inst, out_modes)
            continue
        ports = [p for p in comp.ports() if p.kind != PortKind.ELECTRICAL]
        in_ports = [p for p in ports if p.direction == PortDirection.IN]
        out_ports = [p for p in ports if p.direction == PortDirection.OUT]
        model = comp.detector_model()
        if model is not None:
            for p in in_ports:
                link = in_link.get(f"{inst}.{p.name}")
                if link is None or link.id not in link_modes:
                    continue
                m = link_modes[link.id]
                detector_modes.setdefault(inst, m)
                arr = arrivals.get((inst, p.name))
                radius_of = getattr(comp, "aperture_radius", None)
                radius = radius_of() if radius_of is not None else None
                if radius is not None and arr:
                    t = aperture_transmission(radius, arr[0].mode)
                    if t < 1.0 - 1e-12:
                        _scalar_elements(complex(math.sqrt(t)), m[0], m[1], inst, elements)
                op = np.asarray(model.efficiency(wl), dtype=complex)[0]
                eta = float(np.real(np.trace(op)) / 2.0)
                if not np.allclose(op, eta * eye, atol=1e-9):
                    elements.append(Block(m, _psd_sqrt(op), label=f"{inst} efficiency"))
                break
            continue
        in_modes: list[int] = []
        connected = False
        for p in in_ports:
            link = in_link.get(f"{inst}.{p.name}")
            if link is not None and link.id in link_modes:
                in_modes += list(link_modes[link.id])
                connected = True
            elif len(in_ports) > 1:
                in_modes += [
                    new_mode(f"{inst}.{p.name}:H (vacuum)"),
                    new_mode(f"{inst}.{p.name}:V (vacuum)"),
                ]
            else:
                in_modes += [new_mode(f"{inst}.{p.name}:H"), new_mode(f"{inst}.{p.name}:V")]
        if not connected or not out_ports:
            continue
        m_mat = np.zeros((2 * len(out_ports), 2 * len(in_ports)), dtype=complex)
        out_index = {p.name: k for k, p in enumerate(out_ports)}
        for ci, p in enumerate(in_ports):
            arr = arrivals.get((inst, p.name))
            if not arr:
                continue
            beam = unit(arr[0])
            outs = comp.apply_quality(beam, comp.propagate(p.name, beam, ctx), ctx)
            for out_port, ob in outs:
                if out_port in out_index:
                    co = out_index[out_port]
                    m_mat[2 * co : 2 * co + 2, 2 * ci : 2 * ci + 2] += ob.jones[0] * np.exp(
                        2j * np.pi * ob.opl / lam
                    )
        n_out_modes = 2 * len(out_ports)
        pool = list(in_modes)
        out_flat = pool[:n_out_modes]
        for k in range(len(out_flat), n_out_modes):
            port = out_ports[k // 2].name
            out_flat.append(new_mode(f"{inst}.{port}:{'HV'[k % 2]}"))
        _component_elements(
            comp, inst, m_mat, in_modes, out_flat, len(in_ports), len(out_ports), elements
        )
        out_modes = {
            p.name: (out_flat[2 * k], out_flat[2 * k + 1]) for k, p in enumerate(out_ports)
        }
        emit_links(inst, out_modes)

    n = max(len(labels), 1)
    circuit = Circuit(n)
    try:
        for e in elements:
            circuit.add(e)
    except FockError as exc:
        raise FockCompileError(f"cannot assemble the decomposed circuit: {exc}") from exc
    verified, err, fallback = True, 0.0, False
    if verify:
        det_ids = [d for d in net.components if d in detector_modes]
        in_specs = [f"{i}.{p}" for i, p in starts]
        angles = _fock_angles(fock)
        bases = []
        for i, _p in starts:
            if i in angles:
                c, s = math.cos(angles[i]), math.sin(angles[i])
                bases.append((("pol", "perp"), np.array([[c, -s], [s, c]], dtype=complex)))
            else:
                bases.append(None)
        ref = compile_setup_network(
            setup,
            library,
            lam,
            inputs=in_specs,
            detectors=det_ids,
            bases=bases,
            seed_modes=[(seed_modes or {}).get(sp) for sp in in_specs],
            should_cancel=should_cancel,
            net=net,
        )
        t = circuit.matrix()
        rows = [m for d in det_ids for m in detector_modes[d]]
        cols = [m for s in in_specs for m in input_modes[s]]
        got = t[np.ix_(rows, cols)]
        err = float(np.max(np.abs(got - ref.transfer))) if got.size else 0.0
        verified = err < 1e-8
        if not verified:
            fallback = True
            notes.append(
                f"decomposition differs from the exact compilation by {err:.3g}; the compiled "
                "transfer matrix is given as one block"
            )
            big = Circuit(max(ref.circuit.n_modes, 1))
            for e in ref.circuit.elements:
                big.add(replace(e, label="compiled network") if isinstance(e, Block) else e)
            circuit = big
            input_modes = {s: (2 * k, 2 * k + 1) for k, s in enumerate(in_specs)}
            detector_modes = {d: (2 * k, 2 * k + 1) for k, d in enumerate(det_ids)}
            labels = [str(m) for m in ref.input_modes] + [""] * (big.n_modes - len(ref.input_modes))
    return Extraction(
        circuit,
        input_modes,
        detector_modes,
        labels,
        lam,
        verified,
        err,
        fallback,
        notes,
        _fock_angles(fock),
    )


def _fock_angles(fock: dict) -> dict[str, float]:
    out = {}
    for i, occ in fock.items():
        j = np.asarray(occ[2], dtype=complex)
        th = math.atan2(float(np.real(j[1])), float(np.real(j[0])))
        if abs(th) > 1e-12:
            out[i] = th
    return out


def _component_elements(comp, inst, m_mat, in_modes, out_flat, n_in, n_out, elements) -> None:
    tol = 1e-9
    if comp.type_name == "PhaseShifter" and n_in == 1 and n_out == 1:
        for k in range(2):
            _scalar_one(complex(m_mat[k, k]), in_modes[k], inst, elements)
        return
    if n_in == 1 and n_out == 1:
        if np.allclose(m_mat, m_mat[0, 0] * np.eye(2), atol=tol):
            if not np.isclose(m_mat[0, 0], 1.0, atol=tol):
                _scalar_elements(complex(m_mat[0, 0]), in_modes[0], in_modes[1], inst, elements)
            return
        elements.append(Block((in_modes[0], in_modes[1]), m_mat, label=inst))
        return
    if n_in == 2 and n_out == 2:
        ok = True
        blocks = []
        for k in range(2):
            u = m_mat[np.ix_([k, 2 + k], [k, 2 + k])]
            cross = m_mat[np.ix_([k, 2 + k], [1 - k, 3 - k])]
            if np.max(np.abs(cross)) > tol or not np.allclose(u.conj().T @ u, np.eye(2), atol=1e-7):
                ok = False
                break
            c, s = abs(u[0, 0]), abs(u[0, 1])
            theta = 2.0 * math.atan2(s, c)
            tl = math.atan2(u[0, 0].imag, u[0, 0].real) if c > tol else 0.0
            bl = math.atan2(u[0, 1].imag, u[0, 1].real) - math.pi / 2 if s > tol else 0.0
            br = (
                math.atan2(u[1, 1].imag, u[1, 1].real) - bl
                if c > tol
                else math.atan2(u[1, 0].imag, u[1, 0].real) + math.pi / 2 - tl
            )
            rebuilt = bs_matrix(theta, tl, bl, 0.0, br)
            if not np.allclose(rebuilt, u, atol=1e-7):
                ok = False
                break
            blocks.append(
                BeamSplitter((in_modes[k], in_modes[2 + k]), theta, tl, bl, 0.0, br, "Rx", inst)
            )
        if ok:
            elements.extend(blocks)
            return
    mode_list = list(dict.fromkeys(list(in_modes) + list(out_flat)))
    pos = {m: i for i, m in enumerate(mode_list)}
    k = len(mode_list)
    mat = np.zeros((k, k), dtype=complex)
    for o, om in enumerate(out_flat):
        for i, im in enumerate(in_modes):
            mat[pos[om], pos[im]] += m_mat[o, i]
    elements.append(Block(tuple(mode_list), mat, label=inst))


def extraction_experiment(
    se: SetupExperiment, wavelength: float | None = None, should_cancel=None
) -> tuple[Experiment, Extraction]:
    """Editable circuit-editor experiment equivalent to ``se`` (single wavelength)."""
    channels = se.plan.channels
    ex = extract_circuit(
        se.setup,
        se.library,
        wavelength if wavelength is not None else channels[0].wavelength,
        should_cancel=should_cancel,
        seed_modes={f"{c.inst}.{c.port}": c.mode for c in channels if c.mode is not None},
    )
    n = ex.circuit.n_modes
    plan = se.plan
    if ex.fallback:
        mode_of = lambda c, p: 2 * c + p  # noqa: E731
        by_key = {f"{i}.{p}": k for k, (i, p) in enumerate(plan.keys)}
        order = list(ex.input_modes)
        index = {key: order.index(key) for key in by_key if key in order}
        mode_of = lambda c, p: 2 * index[f"{plan.keys[c][0]}.{plan.keys[c][1]}"] + p  # noqa: E731
    else:

        def mode_of(c: int, p: int) -> int:
            return ex.input_modes[f"{plan.keys[c][0]}.{plan.keys[c][1]}"][p]

    mixture = plan.make(mode_of)
    dets = []
    for k, d in enumerate(se.detector_ids):
        if d in ex.detector_modes:
            dets.append(DetectorSpec(tuple(ex.detector_modes[d]), se.compiled.detectors[k].model))
    exp = Experiment(
        ex.circuit,
        [0] * n,
        detectors=dets,
        mode_labels=list(ex.mode_labels) + [""] * (n - len(ex.mode_labels)),
        name=se.setup.name,
        limits=se.experiment.limits,
        input_mixture=mixture,
        rate_hz=se.rate_hz,
        window_s=se.window_s,
        duration_s=se.duration_s,
    )
    return exp, ex
