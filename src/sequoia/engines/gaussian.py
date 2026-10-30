import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import constants, sparse
from scipy.sparse import csgraph

from sequoia.analysis.metrics import ratio
from sequoia.engines.cache import ResourceLimitError, RunCache, check_cancel, record_warnings
from sequoia.engines.classical import SimulationResult
from sequoia.engines.rates import _Engine, _Source
from sequoia.errors import Issue, SetupError
from sequoia.library.store import Library
from sequoia.light.gaussian import aperture_transmission
from sequoia.light.state import combine
from sequoia.setup.model import Setup

SMALL_COMPONENT = 6
MAX_DENSE_SIDE = 2500
MAX_MODES = 160
SCHMIDT_CUTOFF = 1e-10
MAX_DETECTORS = 6
PNR_GRID_TOL = 1e-10
PNR_MAX_GRID = 4096
PNR_MAX_POINTS = 150_000
WINDOW_RTOL = 1e-9


class IncompatibleWindowsError(SetupError):
    pass


class UnsupportedSourceError(SetupError):
    pass


@dataclass
class _Batch:
    """Independent groups of modes with the same size, as stacked arrays.

    ``u`` (B, 2a, r) and ``v`` (B, 2b, r) are the Schmidt vectors of the signal and idler
    nodes (two polarizations each), ``sigma`` (B, r) the normalised singular values. ``ys`` and
    ``yi`` map every Schmidt mode to the detector amplitude of its node; ``ei`` lists the
    (batch, signal node, idler node) triples that share a frequency.
    """

    u: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    s_nodes: np.ndarray
    i_nodes: np.ndarray
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class _Member:
    weight: float
    batches: list[_Batch]


@dataclass
class GaussianSource:
    inst: str
    pairs_per_window: float
    window: float
    pulsed: bool
    members: list[_Member]
    modes: int


@dataclass
class GaussianResult:
    """Exact photon statistics per time window (one pulse, or 1/dnu for CW sources)."""

    channels: list[str]
    window: float
    clicks: dict[tuple[str, ...], float]
    mean_photons: dict[str, float]
    sources: dict[str, dict[str, Any]]
    issues: list[Issue] = field(default_factory=list)
    analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    pnr: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)
    pnr_info: dict[tuple[str, ...], dict[str, Any]] = field(default_factory=dict)
    gain_scale: float | dict[str, float] = 1.0
    source_scale: dict[str, float] = field(default_factory=dict)

    def probability(self, *names: str) -> float:
        """Probability that all the named channels click in the same window."""
        unknown = [n for n in names if n not in self.channels]
        if unknown:
            raise KeyError(f"unknown channel {unknown[0]!r} (have {self.channels})")
        return self.clicks[tuple(sorted(names, key=self.channels.index))]

    def rate(self, *names: str) -> float:
        return self.probability(*names) / self.window


def _real_rep(c: np.ndarray) -> np.ndarray:
    top = np.concatenate([c.real, -c.imag], axis=-1)
    bottom = np.concatenate([c.imag, c.real], axis=-1)
    return np.concatenate([top, bottom], axis=-2)


def _sigma_normal(sigma: np.ndarray, scale: float) -> np.ndarray:
    """Normal-ordered quadrature covariance of two-mode squeezers (x block, p block)."""
    r = scale * sigma
    n = np.sinh(r) ** 2
    m = np.sinh(r) * np.cosh(r)
    b, k = sigma.shape
    out = np.zeros((b, 4 * k, 4 * k))
    idx = np.arange(k)
    for sign, off in ((1.0, 0), (-1.0, 2 * k)):
        out[:, off + idx, off + idx] = n
        out[:, off + k + idx, off + k + idx] = n
        out[:, off + idx, off + k + idx] = sign * m
        out[:, off + k + idx, off + idx] = sign * m
    return out


def log_generating_function(
    ks: np.ndarray, sigma: np.ndarray, scale: float, lam: np.ndarray
) -> complex:
    """log <prod_d z_d^{N_d}> with lam_d = 1 - z_d, summed over independent blocks.

    ``ks`` is (D, B, 4r, 4r), the real quadrature form of the detector number operators in the
    Schmidt modes of each block. The Gaussian formula is 1/sqrt(det(1 + K Sigma_N)).
    """
    sig = _sigma_normal(sigma, scale)
    k = np.tensordot(lam, ks, axes=(0, 0))
    mat = np.eye(sig.shape[-1])[None] + k @ sig
    sign, logdet = np.linalg.slogdet(mat)
    return complex(-0.5 * np.sum(logdet + np.log(sign.astype(complex))))


def _node_operators(ys: np.ndarray, yi: np.ndarray, pairs) -> np.ndarray:
    """Detector number operator in the Schmidt-mode basis, (B, 2r, 2r)."""
    c_ss = np.einsum("bjpr,bjps->brs", ys.conj(), ys)
    c_ii = np.einsum("bkpr,bkps->brs", yi.conj(), yi)
    batch, j, k = pairs
    c_si = np.zeros_like(c_ss)
    if len(batch):
        prod = np.einsum("npr,nps->nrs", ys[batch, j].conj(), yi[batch, k])
        np.add.at(c_si, batch, prod)
    top = np.concatenate([c_ss, c_si], axis=2)
    bottom = np.concatenate([c_si.conj().transpose(0, 2, 1), c_ii], axis=2)
    return np.concatenate([top, bottom], axis=1)


class _GaussianEngine:
    def __init__(self, engine: _Engine, should_cancel) -> None:
        self.engine = engine
        self.should_cancel = should_cancel
        self.tagged: dict[str, str] = {}
        self.truncated = 0.0
        for ch in engine.channels.values():
            self.tagged.setdefault(ch.detector, ch.name)

    def cancel(self) -> None:
        check_cancel(self.should_cancel)

    def source(self, src: _Source) -> GaussianSource | None:
        eng = self.engine
        if src.exchange:
            eng.issue(
                "warning",
                "gaussian_exchange_unsupported",
                f"'{src.inst}'",
                "signal and idler leave the same port; the Gaussian engine does not model "
                "this source and skips it",
            )
            return None
        ps = src.emission.source
        pulsed = src.pulsed
        window = 1.0 / ps.repetition_rate if pulsed else 1.0 / ps.dnu
        n_s = np.asarray(src.grid["signal"], dtype=np.int64)
        n_i = np.asarray(src.grid["idler"], dtype=np.int64)
        members_amp = np.asarray(src.members)
        norms = np.array([float(np.sum(np.abs(m) ** 2)) for m in members_amp])
        total = float(norms.sum())
        if total <= 0:
            return None
        dets = list(self.tagged)
        h_s = np.stack([src.h[d]["signal"] for d in dets])
        h_i = np.stack([src.h[d]["idler"] for d in dets])
        support = np.abs(members_amp).sum(axis=(0, 1, 2)) > 0
        js, ms = np.nonzero(support)
        ks = src.idx[js, ms]
        s_count, i_count = len(n_s), len(n_i)
        edges_r = [js]
        edges_c = [s_count + ks]
        pos = np.clip(np.searchsorted(n_i, n_s), 0, i_count - 1)
        same = n_i[pos] == n_s
        both = (
            same
            & (np.abs(h_s).sum(axis=(0, 2, 3)) > 0)
            & (np.abs(h_i[:, pos]).sum(axis=(0, 2, 3)) > 0)
        )
        sj = np.flatnonzero(both)
        edges_r.append(sj)
        edges_c.append(s_count + pos[sj])
        rows = np.concatenate(edges_r)
        cols = np.concatenate(edges_c)
        graph = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(s_count + i_count, s_count + i_count)
        )
        used = np.zeros(s_count + i_count, dtype=bool)
        used[rows] = True
        used[cols] = True
        _, labels = csgraph.connected_components(graph, directed=False)
        labels = np.where(used, labels, -1)
        order = np.argsort(labels, kind="stable")
        order = order[labels[order] >= 0]
        label_sorted = labels[order]
        starts = np.flatnonzero(np.r_[True, label_sorted[1:] != label_sorted[:-1]])
        stops = np.r_[starts[1:], len(order)]
        comps = []
        for a, b in zip(starts, stops, strict=True):
            nodes = order[a:b]
            comps.append((nodes[nodes < s_count], nodes[nodes >= s_count] - s_count))
        mu = src.rate * window * total
        members = []
        modes = 0
        for m_index, amp in enumerate(members_amp):
            weight = norms[m_index] / total
            if weight <= 0:
                continue
            self.cancel()
            batches = self.member_batches(
                amp / math.sqrt(norms[m_index]), src, comps, n_s, n_i, h_s, h_i, s_count
            )
            modes += sum(b.sigma.size for b in batches)
            members.append(_Member(weight, batches))
        return GaussianSource(src.inst, mu, window, pulsed, members, modes)

    def member_batches(self, amp, src, comps, n_s, n_i, h_s, h_i, s_count) -> list[_Batch]:
        groups: dict[tuple[int, int], list[int]] = {}
        big = []
        for c, (s_nodes, i_nodes) in enumerate(comps):
            if len(s_nodes) == 0 or len(i_nodes) == 0:
                continue
            if len(s_nodes) + len(i_nodes) <= SMALL_COMPONENT:
                groups.setdefault((len(s_nodes), len(i_nodes)), []).append(c)
            else:
                big.append(c)
        rows_j, cols_m = np.nonzero(np.abs(amp).sum(axis=(0, 1)) > 0)
        k_of = np.full((len(n_s), amp.shape[-1]), -1)
        k_of[rows_j, cols_m] = src.idx[rows_j, cols_m]
        lookup: dict[tuple[int, int], np.ndarray] = {}
        for j, m in zip(rows_j, cols_m, strict=True):
            lookup[(int(j), int(k_of[j, m]))] = amp[:, :, j, m]
        out: list[_Batch] = []
        for (a, b), ids in groups.items():
            s_idx = np.stack([comps[c][0] for c in ids])
            i_idx = np.stack([comps[c][1] for c in ids])
            jmat = np.zeros((len(ids), 2 * a, 2 * b), dtype=complex)
            for bi, (srow, irow) in enumerate(zip(s_idx, i_idx, strict=True)):
                for ja, j in enumerate(srow):
                    for kb, k in enumerate(irow):
                        block = lookup.get((int(j), int(k)))
                        if block is not None:
                            jmat[bi, 2 * ja : 2 * ja + 2, 2 * kb : 2 * kb + 2] = block
            u, s, vh = np.linalg.svd(jmat, full_matrices=False)
            out.append(self.finish(u, s, vh.transpose(0, 2, 1), s_idx, i_idx, n_s, n_i))
        for c in big:
            self.cancel()
            s_nodes, i_nodes = comps[c]
            if max(len(s_nodes), len(i_nodes)) > MAX_DENSE_SIDE:
                raise ResourceLimitError(
                    f"a connected block of the pair amplitude has {len(s_nodes)} x "
                    f"{len(i_nodes)} frequency points (limit {MAX_DENSE_SIDE}); use a coarser "
                    "frequency lattice in the pair source"
                )
            jmat = np.zeros((2 * len(s_nodes), 2 * len(i_nodes)), dtype=complex)
            s_pos = {int(j): n for n, j in enumerate(s_nodes)}
            i_pos = {int(k): n for n, k in enumerate(i_nodes)}
            for (j, k), block in lookup.items():
                if j in s_pos and k in i_pos:
                    jj, kk = s_pos[j], i_pos[k]
                    jmat[2 * jj : 2 * jj + 2, 2 * kk : 2 * kk + 2] = block
            u, s, vh = np.linalg.svd(jmat, full_matrices=False)
            keep = int(np.sum(s > SCHMIDT_CUTOFF * s[0])) if s[0] > 0 else 0
            keep = max(1, min(keep, MAX_MODES))
            lost = float(np.sum(s[keep:] ** 2) / np.sum(s**2)) if s[0] > 0 else 0.0
            self.truncated = max(self.truncated, lost)
            if lost > 1e-4:
                self.engine.issue(
                    "warning",
                    "gaussian_truncated",
                    f"'{src.inst}'",
                    f"the Schmidt decomposition was truncated to {keep} modes, dropping "
                    f"{100 * lost:.3g}% of the pair probability; use a coarser frequency lattice "
                    "or narrower filters",
                )
            out.append(
                self.finish(
                    u[None, :, :keep],
                    s[None, :keep],
                    vh[:keep].T[None],
                    s_nodes[None],
                    i_nodes[None],
                    n_s,
                    n_i,
                )
            )
        norm = sum(float(np.sum(b.sigma**2)) for b in out)
        if norm > 0:
            for b in out:
                b.sigma /= math.sqrt(norm)
        return out

    def finish(self, u, s, v, s_idx, i_idx, n_s, n_i) -> _Batch:
        equal = n_s[s_idx][:, :, None] == n_i[i_idx][:, None, :]
        b, j, k = np.nonzero(equal)
        return _Batch(u, v, s.astype(float), s_idx, i_idx, (b, j, k))


def _detector_forms(batches: list[_Batch], h_s: np.ndarray, h_i: np.ndarray) -> list[np.ndarray]:
    """Real quadrature forms of every detector's number operator, per batch: (D, B, 4r, 4r)."""
    out = []
    for batch in batches:
        a = batch.s_nodes.shape[1]
        b = batch.i_nodes.shape[1]
        u = batch.u.reshape(batch.u.shape[0], a, 2, -1)
        v = batch.v.reshape(batch.v.shape[0], b, 2, -1)
        forms = []
        for d in range(h_s.shape[0]):
            ys = np.einsum("bjpq,bjqr->bjpr", h_s[d][batch.s_nodes], u)
            yi = np.einsum("bkpq,bkqr->bkpr", h_i[d][batch.i_nodes], v)
            forms.append(_real_rep(_node_operators(ys, yi, batch.pairs)))
        out.append(np.stack(forms))
    return out


class GaussianModel:
    def __init__(
        self,
        sources: list[GaussianSource],
        forms,
        channels: list[str],
        dark: np.ndarray,
        leakage: np.ndarray,
        laser_leakage: dict[str, np.ndarray] | None = None,
        pumps: dict[str, list[str]] | None = None,
    ):
        self.sources = sources
        self.forms = forms
        self.channels = channels
        self.dark = dark
        self.leakage = leakage
        self.laser_leakage = laser_leakage
        self.pumps = pumps if pumps is not None else {s.inst: [] for s in sources}

    def factors(self, gain_scale: float | Mapping[str, float]) -> dict[str, float]:
        """Pump multiplier of every pair source; a float applies to all of them."""
        known = set(self.pumps) | {s.inst for s in self.sources}
        if isinstance(gain_scale, Mapping):
            unknown = set(gain_scale) - known
            if unknown:
                raise ValueError(
                    f"unknown pair source(s) {sorted(unknown)} in gain_scale (have {sorted(known)})"
                )
            out = {k: float(gain_scale.get(k, 1.0)) for k in known}
        else:
            out = dict.fromkeys(known, float(gain_scale))
        for k, v in out.items():
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"pump multiplier of '{k}' must be finite and >= 0, got {v}")
        return out

    def laser_factor(self, laser: str, factors: Mapping[str, float]) -> float:
        values = {factors[s] for s, lasers in self.pumps.items() if laser in lasers}
        if len(values) > 1:
            pumped = sorted(s for s, lasers in self.pumps.items() if laser in lasers)
            raise ValueError(
                f"laser '{laser}' pumps {pumped} with different multipliers; its leakage cannot "
                "follow both, give these sources the same multiplier"
            )
        return values.pop() if values else 1.0

    def background(self, gain_scale: float | Mapping[str, float] = 1.0) -> np.ndarray:
        if self.laser_leakage is None:
            return self.dark + self.leakage * float(gain_scale)
        factors = gain_scale if isinstance(gain_scale, Mapping) else self.factors(gain_scale)
        total = self.dark + self.leakage
        for laser, leak in self.laser_leakage.items():
            total = total + leak * (self.laser_factor(laser, factors) - 1.0)
        return total

    @staticmethod
    def _scale(gain_scale, inst: str) -> float:
        return gain_scale[inst] if isinstance(gain_scale, Mapping) else gain_scale

    def mean_photons(self, d: int, gain_scale: float | Mapping[str, float] = 1.0) -> float:
        """Mean detected photon number of detector ``d`` per window (pairs plus background)."""
        total = float(self.background(gain_scale)[d])
        for src in self.sources:
            c = math.sqrt(src.pairs_per_window * self._scale(gain_scale, src.inst))
            for member in src.members:
                for batch, k in zip(member.batches, self.forms[src.inst][id(member)], strict=True):
                    sig = _sigma_normal(batch.sigma, c)
                    total += member.weight * 0.5 * float(np.einsum("bij,bji->", k[d], sig))
        return total

    def no_click(self, lam: np.ndarray, gain_scale: float | Mapping[str, float] = 1.0) -> complex:
        """P(no click) / generating function for per-detector weights ``lam`` = 1 - z."""
        total = 0.0 + 0j
        for src in self.sources:
            c = math.sqrt(src.pairs_per_window * self._scale(gain_scale, src.inst))
            value = 0.0 + 0j
            for member in src.members:
                log_g = 0.0 + 0j
                for batch, k in zip(member.batches, self.forms[src.inst][id(member)], strict=True):
                    log_g += log_generating_function(k, batch.sigma, c, lam)
                value += member.weight * np.exp(log_g)
            total += np.log(value)
        total -= complex(np.dot(lam, self.background(gain_scale)))
        return complex(np.exp(total))


@dataclass
class GaussianSetup:
    """Prepared Gaussian model of a setup; ``evaluate`` is cheap for any pump scale."""

    model: GaussianModel
    channels: list[str]
    window: float
    specs: list[dict[str, Any]]
    issues: list[Issue]
    source_info: dict[str, dict[str, Any]]
    should_cancel: Callable[[], bool] | None = None

    def evaluate(
        self,
        gain_scale: float | Mapping[str, float] = 1.0,
        pnr: list[tuple[str, ...]] | None = None,
        pnr_cutoff: int = 12,
    ) -> GaussianResult:
        """``gain_scale`` multiplies the pump power of every pair source (a float) or of the
        named sources (``{source id: factor}``, others 1). It scales the pair gain of a source and
        the classical leakage of the laser that pumps it; dark counts, Fock-source flux and lasers
        that pump no pair source stay fixed."""
        names, model = self.channels, self.model
        given = gain_scale
        gain_scale = model.factors(gain_scale)
        if model.laser_leakage is None:
            gain_scale = float(given)
        n = len(names)
        no_click: dict[frozenset[int], float] = {}
        for size in range(n + 1):
            for subset in itertools.combinations(range(n), size):
                check_cancel(self.should_cancel)
                lam = np.zeros(n)
                lam[list(subset)] = 1.0
                no_click[frozenset(subset)] = float(model.no_click(lam, gain_scale).real)
        clicks: dict[tuple[str, ...], float] = {}
        for size in range(1, n + 1):
            for subset in itertools.combinations(range(n), size):
                p = 0.0
                for k in range(size + 1):
                    for sub in itertools.combinations(subset, k):
                        p += (-1) ** k * no_click[frozenset(sub)]
                clicks[tuple(names[i] for i in subset)] = max(p, 0.0)
        mean = {name: model.mean_photons(i, gain_scale) for i, name in enumerate(names)}
        pnr_out, pnr_info = {}, {}
        for group in pnr or []:
            probs, meta = _pnr(model, names, group, gain_scale, pnr_cutoff, self.should_cancel)
            pnr_out[tuple(group)] = probs
            pnr_info[tuple(group)] = meta
            if not meta["converged"]:
                self.issues.append(
                    Issue(
                        "warning",
                        "gaussian_pnr_unconverged",
                        ", ".join(group),
                        f"the photon-number distribution did not converge within a "
                        f"{PNR_MAX_GRID}-point contour (grid change {meta['grid_change']:.2g}, "
                        "a numerical convergence estimate, not an error bound); reduce the pump "
                        "multiplier or the mean photon number",
                    )
                )
        info = {
            k: {**v, "pairs_per_window": v["pairs_per_window"] * model._scale(gain_scale, k)}
            for k, v in self.source_info.items()
        }
        result = GaussianResult(
            names,
            self.window,
            clicks,
            mean,
            info,
            list(self.issues),
            gain_scale=dict(given) if isinstance(given, Mapping) else given,
            source_scale={k: model._scale(gain_scale, k) for k in model.pumps},
            pnr=pnr_out,
            pnr_info=pnr_info,
        )
        result.analyses = analyse(result, self.specs)
        return result


def prepare_gaussian(
    setup: Setup,
    library: Library,
    *,
    classical: SimulationResult | None = None,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
    max_memory_bytes: float | None = None,
    fock_as_poisson: bool = False,
    sources: Iterable[str] | None = None,
) -> GaussianSetup:
    engine = _Engine(setup, library, classical, should_cancel, cache, max_memory_bytes)
    engine.discover()
    engine.emit_sources()
    pumps = _pump_lasers(engine)
    if sources is not None:
        chosen = set(sources)
        known = {s.inst for s in engine.sources}
        if chosen - known:
            raise SetupError(
                f"unknown pair source(s) {sorted(chosen - known)}; the setup has {sorted(known)}"
            )
        for src in engine.sources:
            if src.inst not in chosen:
                engine.issue(
                    "warning",
                    "gaussian_source_skipped",
                    f"'{src.inst}'",
                    "not selected for the Gaussian model: its photon pairs are left out; the "
                    "classical light and dark counts at the detectors are kept",
                )
        engine.sources = [s for s in engine.sources if s.inst in chosen]
    engine.transfer_sources()
    engine.cancel_point()
    engine.background()
    ge = _GaussianEngine(engine, should_cancel)
    dets = list(ge.tagged)
    names = [ge.tagged[d] for d in dets]
    if not dets:
        engine.issue(
            "warning",
            "gaussian_no_detectors",
            "setup",
            "no photon detector is connected to a time-tagger channel; connect a detector output "
            "to a tagger channel to get Gaussian statistics",
        )
        return GaussianSetup(
            GaussianModel([], {}, [], np.zeros(0), np.zeros(0)),
            [],
            1.0,
            list(setup.analyses),
            list(engine.issues),
            {},
            should_cancel,
        )
    if len(dets) > MAX_DETECTORS:
        raise ResourceLimitError(f"{len(dets)} tagged detectors; the limit is {MAX_DETECTORS}")
    sources, forms = [], {}
    for src in engine.sources:
        gs = ge.source(src)
        if gs is None:
            continue
        h_s = np.stack([src.h[d]["signal"] for d in dets])
        h_i = np.stack([src.h[d]["idler"] for d in dets])
        forms[gs.inst] = {id(m): _detector_forms(m.batches, h_s, h_i) for m in gs.members}
        sources.append(gs)
    windows = sorted(s.window for s in sources)
    if windows and windows[-1] > windows[0] * (1.0 + WINDOW_RTOL):
        listing = ", ".join(f"'{s.inst}': {s.window:.6g} s" for s in sources)
        raise IncompatibleWindowsError(
            f"the pair sources have different time windows ({listing}); per-window statistics "
            "of independent sources cannot be combined exactly. Select the sources of one window "
            "(sources=..., listed by gaussian_window_groups), give the sources the same "
            "repetition rate or bandwidth, or simulate them in separate setups"
        )
    window = windows[0] if windows else 1.0
    if not windows:
        engine.issue(
            "warning",
            "gaussian_no_pair_source",
            "setup",
            "no pair source reaches a tagged detector; statistics are per 1 s of background "
            "light only",
        )
    fock = {d: engine.det_data[d].get("fock_hz", 0.0) for d in dets}
    if any(v > 0 for v in fock.values()):
        names = ", ".join(f"'{n}'" for n in engine.fock_sources())
        if not fock_as_poisson:
            raise UnsupportedSourceError(
                f"Fock-state source {names} reaches the detectors; this engine would replace "
                "its number statistics by Poisson light. Use the Photon counts tab, or pass "
                "fock_as_poisson=True to accept the approximation explicitly"
            )
        engine.issue(
            "warning",
            "fock_as_poisson",
            names,
            "approximation: the Fock-state light is simulated as Poisson background at its "
            "mean flux (no antibunching, no two-photon interference)",
        )
    dark = np.array([engine.det_data[d]["dark_hz"] * window for d in dets])
    leakage = np.array([(engine.det_data[d]["leakage_hz"] + fock[d]) * window for d in dets])
    laser_leakage = {k: v * window for k, v in _leakage_by_laser(engine, dets).items()}
    model = GaussianModel(sources, forms, names, dark, leakage, laser_leakage, pumps)
    info = {
        s.inst: {
            "pairs_per_window": s.pairs_per_window,
            "window_s": s.window,
            "pulsed": s.pulsed,
            "modes": s.modes,
            "members": len(s.members),
        }
        for s in sources
    }
    return GaussianSetup(
        model, names, window, list(setup.analyses), list(engine.issues), info, should_cancel
    )


def _pump_lasers(engine: _Engine) -> dict[str, list[str]]:
    fock = set(engine.fock_sources())
    out = {}
    for src in engine.sources:
        lasers: set[str] = set()
        for state in engine.classical.links.values():
            if state.dst.split(".")[0] == src.inst:
                lasers |= {b.source for b in state.end_beams if b.source not in fock}
        out[src.inst] = sorted(lasers)
    return out


def _leakage_by_laser(engine: _Engine, dets: list[str]) -> dict[str, np.ndarray]:
    """Classical leakage rate (Hz) of each laser at each detector: the rate engine's total split
    in proportion to each laser's own detected light (lasers add incoherently)."""
    fock = set(engine.fock_sources())
    out: dict[str, np.ndarray] = {}
    for n, det_id in enumerate(dets):
        det = engine.detectors[det_id]
        radius_of = getattr(engine.net.components[det_id], "aperture_radius", None)
        radius = radius_of() if radius_of is not None else None
        raw: dict[str, float] = {}
        for port in det.ports:
            for state in engine.classical.links.values():
                if state.dst != f"{det_id}.{port}":
                    continue
                by_laser: dict[str, list] = {}
                for beam in state.end_beams:
                    if beam.source not in fock:
                        by_laser.setdefault(beam.source, []).append(beam)
                for laser, beams in by_laser.items():
                    light = combine(beams, engine.classical.sources)
                    with record_warnings():
                        eff = np.asarray(det.model.efficiency(light.wavelengths), dtype=complex)
                    energy = constants.h * constants.c / light.wavelengths
                    rate = np.einsum("nab,nba->n", eff, light.coherency).real / energy
                    collected = 1.0 if radius is None else aperture_transmission(radius, light.mode)
                    raw[laser] = raw.get(laser, 0.0) + float(np.sum(rate)) * collected
        total = sum(raw.values())
        if total <= 0:
            continue
        leak = engine.det_data[det_id]["leakage_hz"]
        for laser, value in raw.items():
            out.setdefault(laser, np.zeros(len(dets)))[n] = leak * value / total
    return out


def gaussian_window_groups(
    setup: Setup,
    library: Library,
    *,
    classical: SimulationResult | None = None,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[tuple[float, list[str]]]:
    """Pair sources grouped by their statistics window (one pulse period, or 1/dnu for CW).

    Only the emission stage runs (no pair propagation). The sources of one group can be
    simulated together with ``prepare_gaussian(..., sources=ids)``.
    """
    engine = _Engine(setup, library, classical, should_cancel, cache)
    engine.emit_sources()
    groups: list[tuple[float, list[str]]] = []
    for src in engine.sources:
        if src.exchange:
            continue
        ps = src.emission.source
        window = 1.0 / ps.repetition_rate if src.pulsed else 1.0 / ps.dnu
        for w, ids in groups:
            if abs(window - w) <= WINDOW_RTOL * w:
                ids.append(src.inst)
                break
        else:
            groups.append((window, [src.inst]))
    return sorted(groups)


def simulate_gaussian(
    setup: Setup,
    library: Library,
    *,
    classical: SimulationResult | None = None,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
    gain_scale: float | Mapping[str, float] = 1.0,
    pnr: list[tuple[str, ...]] | None = None,
    pnr_cutoff: int = 12,
    max_memory_bytes: float | None = None,
    fock_as_poisson: bool = False,
    sources: Iterable[str] | None = None,
) -> GaussianResult:
    """Exact multimode photon-number statistics of the pair sources at the detectors.

    The pair amplitude of every source is decomposed into independent two-mode squeezers
    (Schmidt modes; connected frequency blocks, so CW sources cost one 2x2 problem per
    frequency pair). The detectors' efficiency and path operators of the rate engine act as
    loss and linear optics; threshold-click probabilities and photon-number distributions follow
    from the Gaussian generating function. ``gain_scale`` multiplies the pump power (the
    squeezing parameter squared) of all sources or, as ``{source id: factor}``, of each source;
    1 reproduces the rate engine's pair rate at low gain (see ``GaussianSetup.evaluate``).
    ``sources`` restricts the model to those pair sources (one window group).
    Statistics are per window: one pulse, or 1/dnu for continuous sources. Dark counts and
    classical leakage are added as Poisson noise; timing jitter and dead time are not modelled.
    """
    prepared = prepare_gaussian(
        setup,
        library,
        classical=classical,
        cache=cache,
        should_cancel=should_cancel,
        max_memory_bytes=max_memory_bytes,
        fock_as_poisson=fock_as_poisson,
        sources=sources,
    )
    return prepared.evaluate(gain_scale, pnr, pnr_cutoff)


def _pnr(model, names, group, gain_scale, cutoff, should_cancel=None):
    idx = [names.index(g) for g in group]
    size = cutoff + 1

    def contour(point_lengths, indices):
        shape = tuple(point_lengths)
        values = np.zeros(shape, dtype=complex)
        grids = [np.exp(2j * np.pi * np.arange(n) / n) for n in shape]
        for point in np.ndindex(*shape):
            check_cancel(should_cancel)
            lam = np.zeros(len(names), dtype=complex)
            for slot, d in enumerate(indices):
                lam[d] = 1.0 - grids[slot][point[slot]]
            values[point] = model.no_click(lam, gain_scale)
        return np.clip((np.fft.fftn(values) / values.size).real, 0.0, None)

    lengths, bounds, converged = [], [], True
    for d in idx:
        length = max(16, 2 * size)
        previous = contour([length], [d])
        while True:
            doubled = contour([2 * length], [d])
            diff = float(np.max(np.abs(doubled[:size] - previous[:size])))
            if diff < PNR_GRID_TOL:
                break
            if 2 * length >= PNR_MAX_GRID:
                converged = False
                length = 2 * length
                break
            length, previous = 2 * length, doubled
        lengths.append(length)
        bounds.append(diff)
    points = int(np.prod(lengths, dtype=np.int64))
    if len(idx) > 1 and points > PNR_MAX_POINTS:
        raise ResourceLimitError(
            f"the joint photon-number distribution needs {points} generating-function "
            f"evaluations (limit {PNR_MAX_POINTS}); lower the pump multiplier, the cutoff or "
            "the number of channels"
        )
    probs = contour(lengths, idx)
    view = tuple(slice(0, size) for _ in idx)
    kept = probs[view]
    kept = np.pad(kept, [(0, size - n) for n in kept.shape])
    retained = float(kept.sum())
    meta = {
        "cutoff": cutoff,
        "retained": retained,
        "tail": max(1.0 - retained, 0.0),
        "grid": tuple(int(n) for n in lengths),
        # largest change of a retained bin when the contour doubles: convergence evidence only
        "grid_change": float(max(bounds)) if bounds else 0.0,
        "converged": converged,
    }
    return kept, meta


def detector_response(
    result: GaussianResult, group: tuple[str, ...], detectors: dict[str, Any]
) -> np.ndarray:
    """Click-count distribution of a PNR group seen through multiplexed or threshold detectors.

    ``detectors`` maps a channel name to a ``sequoia.fock.detectors`` model. The Gaussian result
    already contains each detector's efficiency and dark counts, so the models must be ideal
    (efficiency 1, no dark clicks); only their response matrix D(k|n) is applied.
    """
    probs = np.array(result.pnr[tuple(group)], dtype=float)
    for axis, name in enumerate(group):
        det = detectors.get(name)
        if det is None:
            continue
        if det.efficiency != 1.0 or det.dark != 0.0:
            raise ValueError(
                f"detector model for {name} has efficiency {det.efficiency} / dark {det.dark}; "
                "efficiency and dark counts are already part of the Gaussian statistics"
            )
        matrix = det.response_matrix(probs.shape[axis] - 1)
        probs = np.moveaxis(np.tensordot(probs, matrix, axes=([axis], [0])), -1, axis)
    return probs


def analyse(result: GaussianResult, specs) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    w = result.window
    for spec in specs:
        name = str(spec.get("name", spec.get("type", "analysis")))
        kind = spec.get("type")
        try:
            if kind == "coincidences":
                a, b = spec["channels"]
                pa, pb = result.probability(a), result.probability(b)
                pab = result.probability(a, b)
                acc = pa * pb
                out[name] = {
                    "type": "coincidences",
                    "channels": [a, b],
                    "window_s": w,
                    "singles_a_hz": pa / w,
                    "singles_b_hz": pb / w,
                    "coincidence_rate_hz": pab / w,
                    "accidental_hz": acc / w,
                    "true_hz": (pab - acc) / w,
                    "car": ratio(pab - acc, acc),
                    "car_total": ratio(pab, acc),
                    "heralding_a_given_b": (pab / pb) if pb > 0 else 0.0,
                    "heralding_b_given_a": (pab / pa) if pa > 0 else 0.0,
                }
            elif kind == "heralded_g2":
                h, a, b = spec["herald"], spec["a"], spec["b"]
                ph, pha, phb = (
                    result.probability(h),
                    result.probability(h, a),
                    result.probability(h, b),
                )
                phab = result.probability(h, a, b)
                g2 = phab * ph / (pha * phb) if pha > 0 and phb > 0 else math.nan
                out[name] = {
                    "type": "heralded_g2",
                    "herald": h,
                    "a": a,
                    "b": b,
                    "window_s": w,
                    "triples_hz": phab / w,
                    "g2_heralded": g2,
                }
            elif kind in ("g2", "histogram"):
                a, b = spec["channels"]
                pa, pb = result.probability(a), result.probability(b)
                pab = result.probability(a, b)
                out[name] = {
                    "type": "g2",
                    "channels": [a, b],
                    "window_s": w,
                    "g2_at_center": pab / (pa * pb) if pa * pb > 0 else math.nan,
                }
        except (KeyError, ValueError):
            continue
    return out
