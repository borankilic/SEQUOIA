import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from sequoia.fock.circuit import Circuit, FockError, Loss
from sequoia.fock.engine import Distribution, State


def _binomial_thinning(n_max: int, eta: float) -> np.ndarray:
    m = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        for j in range(n + 1):
            m[n, j] = math.comb(n, j) * eta**j * (1.0 - eta) ** (n - j)
    return m


def _with_dark(matrix: np.ndarray, dark: float, saturate_at: int) -> np.ndarray:
    if dark <= 0.0:
        return matrix
    out = (1.0 - dark) * matrix
    for k in range(matrix.shape[1]):
        out[:, min(k + 1, saturate_at)] += dark * matrix[:, k]
    return out


@dataclass(frozen=True)
class Detector:
    efficiency: float = 1.0
    dark: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.efficiency <= 1.0:
            raise FockError(f"detector efficiency must be in [0, 1], got {self.efficiency}")
        if not 0.0 <= self.dark <= 1.0:
            raise FockError(f"dark click probability must be in [0, 1], got {self.dark}")

    kind = "detector"

    def response_matrix(self, n_max: int) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"kind": self.kind, "efficiency": self.efficiency, "dark": self.dark}


@dataclass(frozen=True)
class IdealPNR(Detector):
    kind = "pnr"

    def response_matrix(self, n_max: int) -> np.ndarray:
        m = _binomial_thinning(n_max, self.efficiency)
        return _with_dark(m, self.dark, n_max)


@dataclass(frozen=True)
class Threshold(Detector):
    kind = "threshold"

    def response_matrix(self, n_max: int) -> np.ndarray:
        m = np.zeros((n_max + 1, 2))
        for n in range(n_max + 1):
            click = 1.0 - (1.0 - self.efficiency) ** n
            click = 1.0 - (1.0 - click) * (1.0 - self.dark)
            m[n] = (1.0 - click, click)
        return m


@dataclass(frozen=True, kw_only=True)
class PseudoPNR(Detector):
    """Multiplexed threshold detector: each photon goes to wire w with probability weights[w]
    (equal by default); a wire clicks when it holds at least one detected photon. A photon is
    detected with probability ``efficiency * wire_efficiencies[w]``."""

    wires: int = 4
    weights: tuple[float, ...] | None = None
    wire_efficiencies: tuple[float, ...] | None = None
    kind = "ppnr"

    def __post_init__(self) -> None:
        Detector.__post_init__(self)
        if not 1 <= self.wires <= 14:
            raise FockError("PseudoPNR supports 1 to 14 wires")
        for name in ("weights", "wire_efficiencies"):
            v = getattr(self, name)
            if v is not None and len(v) != self.wires:
                raise FockError(f"{name} must have one entry per wire ({self.wires})")
        if self.weights is not None and abs(sum(self.weights) - 1.0) > 1e-9:
            raise FockError("wire weights must sum to 1")

    @classmethod
    def layered(cls, layers: int, reflectivity: float, **kw) -> "PseudoPNR":
        w = np.array([1.0])
        for _ in range(layers):
            w = np.concatenate([w * reflectivity, w * (1.0 - reflectivity)])
        return cls(wires=2**layers, weights=tuple(float(x) for x in w), **kw)

    def response_matrix(self, n_max: int) -> np.ndarray:
        nw = self.wires
        w = np.full(nw, 1.0 / nw) if self.weights is None else np.asarray(self.weights, float)
        eff = np.ones(nw) if self.wire_efficiencies is None else np.asarray(self.wire_efficiencies)
        miss = 1.0 - self.efficiency * eff
        subsets = np.arange(1 << nw)
        member = ((subsets[:, None] >> np.arange(nw)) & 1).astype(float)
        base = 1.0 - member @ w + member @ (w * miss)
        popcount = member.sum(axis=1).astype(int)
        m = np.zeros((n_max + 1, nw + 1))
        for n in range(n_max + 1):
            q = base**n
            exact = _exact_silent(q, nw)
            clicks = nw - popcount
            for k in range(nw + 1):
                m[n, k] = exact[clicks == k].sum()
        m = np.clip(m, 0.0, 1.0)
        m /= m.sum(axis=1, keepdims=True)
        keep = min(n_max, nw) + 1
        m = m[:, :keep]
        return _with_dark(m, self.dark, keep - 1)

    def to_dict(self) -> dict:
        d = Detector.to_dict(self)
        d["wires"] = self.wires
        if self.weights is not None:
            d["weights"] = list(self.weights)
        if self.wire_efficiencies is not None:
            d["wire_efficiencies"] = list(self.wire_efficiencies)
        return d


def _exact_silent(q: np.ndarray, nw: int) -> np.ndarray:
    # q[U] = P(all wires of U silent, others unrestricted); P(silent set is exactly U)
    exact = q.copy()
    idx = np.arange(1 << nw)
    for b in range(nw):
        has = (idx >> b) & 1 == 0
        exact[idx[has]] -= exact[idx[has] | (1 << b)]
    return exact


def detector_from_dict(d: dict) -> Detector:
    kind = d.get("kind", "pnr")
    kw = {"efficiency": float(d.get("efficiency", 1.0)), "dark": float(d.get("dark", 0.0))}
    if kind == "pnr":
        return IdealPNR(**kw)
    if kind == "threshold":
        return Threshold(**kw)
    if kind == "ppnr":
        tup = lambda v: None if v is None else tuple(float(x) for x in v)  # noqa: E731
        return PseudoPNR(
            wires=int(d.get("wires", 4)),
            weights=tup(d.get("weights")),
            wire_efficiencies=tup(d.get("wire_efficiencies")),
            **kw,
        )
    raise FockError(f"unknown detector kind {kind!r} (pnr, threshold, ppnr)")


@dataclass
class DetectorLayout:
    """Detectors act on groups of output modes (the summed photon number of the group);
    modes in no group are read by an ideal PNR. The detected state has one entry per group,
    ordered by the first mode of the group."""

    groups: list[tuple[tuple[int, ...], Detector]]

    @classmethod
    def from_modes(cls, n_modes: int, models: dict[int, Detector] | None = None):
        models = models or {}
        return cls([((m,), models.get(m, IdealPNR())) for m in range(n_modes)])

    @classmethod
    def build(cls, n_modes: int, groups: Sequence[tuple[Sequence[int], Detector]]):
        seen: set[int] = set()
        out: list[tuple[tuple[int, ...], Detector]] = []
        for modes, det in groups:
            modes = tuple(int(m) for m in modes)
            if not modes or any(not 0 <= m < n_modes for m in modes) or seen & set(modes):
                raise FockError(f"invalid detector group {modes}")
            seen |= set(modes)
            out.append((modes, det))
        for m in range(n_modes):
            if m not in seen:
                out.append(((m,), IdealPNR()))
        out.sort(key=lambda g: min(g[0]))
        return cls(out)

    def check_efficiency(self, circuit: Circuit) -> None:
        in_circuit = {
            e.mode for e in circuit.elements if isinstance(e, Loss) and e.kind == "detector"
        }
        for modes, det in self.groups:
            if det.efficiency < 1.0 and in_circuit & set(modes):
                raise FockError(
                    f"detector on modes {modes} has efficiency {det.efficiency} and the circuit "
                    "also carries a detector-efficiency loss on the same mode; declare it once"
                )

    def response_matrices(self, n_max: int) -> list[np.ndarray]:
        return [det.response_matrix(n_max) for _, det in self.groups]

    def apply(self, dist: Distribution, prune: float = 0.0) -> Distribution:
        n_max = max((sum(s) for s in dist.probs), default=0)
        mats = self.response_matrices(n_max)
        out: dict[State, float] = {}
        for s, p in dist.probs.items():
            rows = []
            for (modes, _), m in zip(self.groups, mats, strict=True):
                n = sum(s[i] for i in modes)
                r = m[n]
                rows.append([(k, v) for k, v in enumerate(r) if v > 0.0])
            for combo in product(*rows):
                w = p
                for _, v in combo:
                    w *= v
                if w > prune:
                    key = tuple(k for k, _ in combo)
                    out[key] = out.get(key, 0.0) + w
        info = dict(dist.info)
        info["detected_modes"] = len(self.groups)
        return Distribution(dict(sorted(out.items())), len(self.groups), info)
