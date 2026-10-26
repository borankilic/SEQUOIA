import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import constants, sparse

from sequoia.components.base import Component
from sequoia.engines.cache import RunCache, check_cancel
from sequoia.engines.classical import build_network
from sequoia.fock.circuit import FockError
from sequoia.fock.compile import FockCompileError
from sequoia.fock.engine import Limits
from sequoia.fock.inputs import (
    InputMixture,
    PureState,
    Term,
    pair_number_mixture,
    pair_number_pmf,
    product_mixture,
)
from sequoia.library.store import Library
from sequoia.light.gaussian import GaussianMode
from sequoia.setup.model import Setup

TAIL = 1e-4
MERGE_RTOL = 1e-3
PRIVATE_OFFSET = 5000
SOURCE_TAG_STRIDE = 10_000
MAX_N = 12
WINDOW_RTOL = 1e-9


class IncompatibleWindowsError(FockError):
    pass


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
