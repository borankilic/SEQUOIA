import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product

import numpy as np

from sequoia.engines.cache import Cancelled
from sequoia.fock.circuit import Circuit, FockError

State = tuple[int, ...]
Photon = tuple[int, Hashable]


class FockResourceError(FockError):
    pass


@dataclass(frozen=True)
class Limits:
    max_photons: int = 10
    max_work: float = 2e9


def _check(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise Cancelled("fock calculation cancelled")


@dataclass
class Distribution:
    probs: dict[State, float]
    n_modes: int
    info: dict = field(default_factory=dict)

    @property
    def mass(self) -> float:
        return float(sum(self.probs.values()))

    def by_photon_number(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for s, p in self.probs.items():
            out[sum(s)] = out.get(sum(s), 0.0) + p
        return dict(sorted(out.items()))

    def lost_mass(self, n_input: int) -> float:
        return float(sum(p for s, p in self.probs.items() if sum(s) < n_input))


_DELTAS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _deltas(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _DELTAS:
        k = 1 << (n - 1)
        idx = np.arange(k)
        d = np.ones((k, n))
        for j in range(1, n):
            d[:, j] = 1.0 - 2.0 * ((idx >> (j - 1)) & 1)
        _DELTAS[n] = (d, np.prod(d, axis=1))
    return _DELTAS[n]


def permanents(a: np.ndarray, should_cancel: Callable[[], bool] | None = None) -> np.ndarray:
    """Glynn formula over a batch of square matrices, shape (B, N, N)."""
    b, n, _ = a.shape
    if n == 0:
        return np.ones(b, dtype=complex)
    if n == 1:
        return a[:, 0, 0].astype(complex)
    d, w = _deltas(n)
    out = np.empty(b, dtype=complex)
    chunk = max(1, int(4e6 // (d.shape[0] * n)))
    for lo in range(0, b, chunk):
        _check(should_cancel)
        rows = np.einsum("di,bij->bdj", d, a[lo : lo + chunk])
        out[lo : lo + chunk] = (np.prod(rows, axis=2) * w).sum(axis=1) / d.shape[0]
    return out


_FACT = np.array([math.factorial(i) for i in range(32)], dtype=float)


def _factorials(rows: np.ndarray, width: int) -> np.ndarray:
    if rows.shape[1] == 0:
        return np.ones(rows.shape[0])
    return np.prod(_FACT[_counts(rows, width)], axis=1)


def _multisets(width: int, total: int) -> np.ndarray:
    if total == 0:
        return np.zeros((1, 0), dtype=np.int64)
    combos = list(combinations_with_replacement(range(width), total))
    return np.asarray(combos, dtype=np.int64)


def _counts(rows: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((rows.shape[0], width), dtype=np.int64)
    for j in range(rows.shape[1]):
        np.add.at(out, (np.arange(rows.shape[0]), rows[:, j]), 1)
    return out


def _coherent_class(
    isometry: np.ndarray,
    n_sys: int,
    in_modes: Sequence[int],
    limits: Limits,
    should_cancel: Callable[[], bool] | None,
    stats: dict,
) -> dict[State, float]:
    n_in = len(in_modes)
    env = isometry.shape[0] - n_sys
    if n_in == 0:
        return {(0,) * n_sys: 1.0}
    cols = np.asarray(sorted(in_modes))
    in_fact = float(np.prod([math.factorial(c) for c in np.bincount(cols)]))
    result: dict[State, float] = {}
    for k in range(0, n_in + 1):
        if k > 0 and env == 0:
            break
        sys_rows = _multisets(n_sys, n_in - k)
        env_rows = _multisets(env, k) if k else np.zeros((1, 0), dtype=np.int64)
        work = sys_rows.shape[0] * env_rows.shape[0] * (2 ** max(n_in - 1, 0)) * max(n_in, 1)
        stats["work"] += work
        if stats["work"] > limits.max_work:
            raise FockResourceError(
                f"estimated work {stats['work']:.3g} exceeds the limit {limits.max_work:.3g} "
                f"({n_in} photons, {n_sys} modes, {env} loss modes); reduce photons or modes, "
                "simplify the circuit or use the Perceval backend"
            )
        sys_f = _factorials(sys_rows, n_sys)
        env_f = _factorials(env_rows, max(env, 1))
        weights = 1.0 / (sys_f[:, None] * env_f[None, :] * in_fact)
        s_idx = np.repeat(np.arange(sys_rows.shape[0]), env_rows.shape[0])
        e_idx = np.tile(np.arange(env_rows.shape[0]), sys_rows.shape[0])
        rows = np.concatenate([sys_rows[s_idx], env_rows[e_idx] + n_sys], axis=1)
        a = isometry[rows[:, :, None], cols[None, None, :]]
        p = np.abs(permanents(a, should_cancel)) ** 2
        stats["permanents"] += rows.shape[0]
        p = (p.reshape(sys_rows.shape[0], env_rows.shape[0]) * weights).sum(axis=1)
        pats = _counts(sys_rows, n_sys)
        for pat, val in zip(map(tuple, pats.tolist()), p.tolist(), strict=True):
            if val > 1e-18:
                result[pat] = result.get(pat, 0.0) + val
    return result


def transfer_isometry(t: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Stack T over an environment block E with E^dag E = 1 - T^dag T, so the dilation is exact."""
    n = t.shape[0]
    defect = np.eye(n) - t.conj().T @ t
    w, v = np.linalg.eigh((defect + defect.conj().T) / 2.0)
    if w.min() < -1e-8:
        raise FockError("transfer matrix is not contractive")
    keep = w > tol
    if not keep.any():
        return t
    env = np.sqrt(w[keep])[:, None] * v[:, keep].conj().T
    return np.vstack([t, env])


def convolve(a: dict[State, float], b: dict[State, float]) -> dict[State, float]:
    out: dict[State, float] = {}
    for sa, pa in a.items():
        for sb, pb in b.items():
            s = tuple(x + y for x, y in zip(sa, sb, strict=True))
            out[s] = out.get(s, 0.0) + pa * pb
    return out


def output_distribution(
    transfer: Circuit | np.ndarray,
    photons: Sequence[Photon],
    *,
    limits: Limits | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Distribution:
    """Photon-number distribution at the output modes.

    ``photons`` lists (input mode, tag); photons with equal tags interfere, photons with
    different tags are mutually incoherent. Lost photons appear as states with fewer photons.
    """
    limits = limits or Limits()
    t = transfer.matrix() if isinstance(transfer, Circuit) else np.asarray(transfer, dtype=complex)
    n = t.shape[0]
    if t.shape != (n, n):
        raise FockError("transfer matrix must be square")
    for m, _ in photons:
        if not 0 <= m < n:
            raise FockError(f"input mode {m} outside 0..{n - 1}")
    if len(photons) > limits.max_photons:
        raise FockResourceError(
            f"{len(photons)} photons exceed the limit of {limits.max_photons}; "
            "raise Limits.max_photons or use the Perceval backend"
        )
    iso = transfer_isometry(t)
    stats = {"work": 0.0, "permanents": 0}
    groups: dict[Hashable, list[int]] = {}
    for m, tag in photons:
        groups.setdefault(tag, []).append(m)
    dist: dict[State, float] = {(0,) * n: 1.0}
    for tag in sorted(groups, key=repr):
        _check(should_cancel)
        cls = _coherent_class(iso, n, groups[tag], limits, should_cancel, stats)
        dist = convolve(dist, cls)
    return Distribution(
        dict(sorted(dist.items())),
        n,
        {
            "backend": "native",
            "photons": len(photons),
            "classes": len(groups),
            "permanents": stats["permanents"],
            "work": stats["work"],
            "loss_modes": iso.shape[0] - n,
        },
    )


def _canonical(photons: list[Photon]) -> tuple:
    relabel: dict[Hashable, int] = {}
    out = []
    for m, tag in sorted(photons, key=lambda x: (x[0], repr(x[1]))):
        relabel.setdefault(tag, len(relabel))
        out.append((m, relabel[tag]))
    return tuple(out)


def source_mixture(
    photons: Sequence[Photon], brightness: float = 1.0, indistinguishability: float = 1.0
) -> list[tuple[float, list[Photon]]]:
    """Each photon is emitted with probability ``brightness``; an emitted photon is in the
    common mode with probability sqrt(indistinguishability), otherwise it carries a private tag,
    so two emitted photons overlap with probability ``indistinguishability`` (Perceval's model)."""
    if not (0.0 <= brightness <= 1.0 and 0.0 <= indistinguishability <= 1.0):
        raise FockError("brightness and indistinguishability must lie in [0, 1]")
    if brightness == 1.0 and indistinguishability == 1.0:
        return [(1.0, list(photons))]
    p_main = math.sqrt(indistinguishability)
    options = [
        (1.0 - brightness, None),
        (brightness * p_main, "main"),
        (brightness * (1.0 - p_main), "own"),
    ]
    merged: dict[tuple, float] = {}
    for combo in product(range(3), repeat=len(photons)):
        w = 1.0
        chosen: list[Photon] = []
        for i, (c, (m, tag)) in enumerate(zip(combo, photons, strict=True)):
            w *= options[c][0]
            if c == 1:
                chosen.append((m, tag))
            elif c == 2:
                chosen.append((m, ("_own", i)))
        if w > 0.0:
            key = _canonical(chosen)
            merged[key] = merged.get(key, 0.0) + w
    return [(w, [(m, tag) for m, tag in key]) for key, w in merged.items()]


def mixed_distribution(
    transfer: Circuit | np.ndarray,
    photons: Sequence[Photon],
    *,
    brightness: float = 1.0,
    indistinguishability: float = 1.0,
    limits: Limits | None = None,
    should_cancel: Callable[[], bool] | None = None,
    backend: Callable[[np.ndarray, Sequence[Photon]], Distribution] | None = None,
) -> Distribution:
    t = transfer.matrix() if isinstance(transfer, Circuit) else np.asarray(transfer, dtype=complex)
    total: dict[State, float] = {}
    info: dict = {}
    for w, ph in source_mixture(photons, brightness, indistinguishability):
        _check(should_cancel)
        if backend is None:
            d = output_distribution(t, ph, limits=limits, should_cancel=should_cancel)
        else:
            d = backend(t, ph)
        for k in ("permanents", "work"):
            info[k] = info.get(k, 0) + d.info.get(k, 0)
        info.update({k: v for k, v in d.info.items() if k not in ("permanents", "work")})
        for s, p in d.probs.items():
            total[s] = total.get(s, 0.0) + w * p
    info["photons"] = len(photons)
    return Distribution(dict(sorted(total.items())), t.shape[0], info)
