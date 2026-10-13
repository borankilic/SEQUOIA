import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sequoia.errors import SequoiaError


class FockError(SequoiaError):
    pass


_BS_TEMPLATES = {
    "Rx": np.array([[1, 1j], [1j, 1]], dtype=complex),
    "Ry": np.array([[1, -1], [1, 1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex),
}


def db_to_loss(db: float) -> float:
    return 1.0 - 10.0 ** (-float(db) / 10.0)


def bs_matrix(theta, phi_tl=0.0, phi_bl=0.0, phi_tr=0.0, phi_br=0.0, convention="Rx"):
    if convention not in _BS_TEMPLATES:
        raise FockError(f"unknown beam splitter convention {convention!r} (Rx, Ry or H)")
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    m = _BS_TEMPLATES[convention].copy()
    m[0, 0] *= np.exp(1j * (phi_tl + phi_tr)) * c
    m[0, 1] *= np.exp(1j * (phi_tr + phi_bl)) * s
    m[1, 0] *= np.exp(1j * (phi_tl + phi_br)) * s
    m[1, 1] *= np.exp(1j * (phi_bl + phi_br)) * c
    return m


@dataclass(frozen=True)
class BeamSplitter:
    modes: tuple[int, int]
    theta: float = math.pi / 2
    phi_tl: float = 0.0
    phi_bl: float = 0.0
    phi_tr: float = 0.0
    phi_br: float = 0.0
    convention: str = "Rx"

    def matrix(self) -> np.ndarray:
        return bs_matrix(
            self.theta, self.phi_tl, self.phi_bl, self.phi_tr, self.phi_br, self.convention
        )


@dataclass(frozen=True)
class PhaseShifter:
    mode: int
    phase: float


@dataclass(frozen=True)
class Permutation:
    modes: tuple[int, ...]
    perm: tuple[int, ...]


@dataclass(frozen=True)
class Loss:
    mode: int
    loss: float
    kind: str = "optical"

    def __post_init__(self) -> None:
        if not 0.0 <= self.loss <= 1.0:
            raise FockError(f"loss probability must be in [0, 1], got {self.loss}")


@dataclass(eq=False)
class Block:
    modes: tuple[int, ...]
    matrix: np.ndarray
    kind: str = "block"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=complex)
        k = len(self.modes)
        if self.matrix.shape != (k, k):
            raise FockError(f"block matrix must be {k}x{k}, got {self.matrix.shape}")
        if np.linalg.norm(self.matrix, 2) > 1.0 + 1e-9:
            raise FockError("block matrix is not contractive (largest singular value > 1)")


Element = BeamSplitter | PhaseShifter | Permutation | Loss | Block


def _element_to_dict(e: Element) -> dict[str, Any]:
    if isinstance(e, BeamSplitter):
        return {
            "type": "beam_splitter",
            "modes": list(e.modes),
            "theta": e.theta,
            "phi_tl": e.phi_tl,
            "phi_bl": e.phi_bl,
            "phi_tr": e.phi_tr,
            "phi_br": e.phi_br,
            "convention": e.convention,
        }
    if isinstance(e, PhaseShifter):
        return {"type": "phase_shifter", "mode": e.mode, "phase": e.phase}
    if isinstance(e, Permutation):
        return {"type": "permutation", "modes": list(e.modes), "perm": list(e.perm)}
    if isinstance(e, Loss):
        return {"type": "loss", "mode": e.mode, "loss": e.loss, "kind": e.kind}
    return {
        "type": "block",
        "modes": list(e.modes),
        "matrix": [[[float(z.real), float(z.imag)] for z in row] for row in e.matrix],
        "kind": e.kind,
    }


def _element_from_dict(d: dict[str, Any]) -> Element:
    t = d.get("type")
    try:
        if t == "beam_splitter":
            return BeamSplitter(
                tuple(d["modes"]),
                float(d.get("theta", math.pi / 2)),
                float(d.get("phi_tl", 0.0)),
                float(d.get("phi_bl", 0.0)),
                float(d.get("phi_tr", 0.0)),
                float(d.get("phi_br", 0.0)),
                d.get("convention", "Rx"),
            )
        if t == "phase_shifter":
            return PhaseShifter(int(d["mode"]), float(d["phase"]))
        if t == "permutation":
            return Permutation(tuple(d["modes"]), tuple(d["perm"]))
        if t == "loss":
            return Loss(int(d["mode"]), float(d["loss"]), d.get("kind", "optical"))
        if t == "block":
            m = np.array([[complex(z[0], z[1]) for z in row] for row in d["matrix"]])
            return Block(tuple(d["modes"]), m, d.get("kind", "block"))
    except (KeyError, TypeError, ValueError) as exc:
        raise FockError(f"invalid circuit element {d!r}: {exc}") from exc
    raise FockError(f"unknown circuit element type {t!r}")


@dataclass
class Circuit:
    n_modes: int
    elements: list[Element] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise FockError("a circuit needs at least one mode")

    def _check_modes(self, modes) -> None:
        for m in modes:
            if not 0 <= m < self.n_modes:
                raise FockError(f"mode {m} outside 0..{self.n_modes - 1}")
        if len(set(modes)) != len(modes):
            raise FockError(f"repeated mode in {tuple(modes)}")

    def add(self, element: Element) -> "Circuit":
        modes = getattr(element, "modes", None)
        self._check_modes(modes if modes is not None else (element.mode,))
        if isinstance(element, Permutation) and sorted(element.perm) != list(
            range(len(element.modes))
        ):
            raise FockError(f"{element.perm} is not a permutation of 0..{len(element.modes) - 1}")
        self.elements.append(element)
        return self

    def beam_splitter(self, a: int, b: int, theta: float = math.pi / 2, **kw) -> "Circuit":
        return self.add(BeamSplitter((a, b), theta, **kw))

    def phase_shifter(self, mode: int, phase: float) -> "Circuit":
        return self.add(PhaseShifter(mode, phase))

    def permutation(self, modes, perm) -> "Circuit":
        return self.add(Permutation(tuple(modes), tuple(perm)))

    def loss(self, mode: int, loss: float, kind: str = "optical") -> "Circuit":
        return self.add(Loss(mode, loss, kind))

    def block(self, modes, matrix) -> "Circuit":
        return self.add(Block(tuple(modes), matrix))

    def matrix(self) -> np.ndarray:
        t = np.eye(self.n_modes, dtype=complex)
        for e in self.elements:
            if isinstance(e, BeamSplitter):
                a, b = e.modes
                t[[a, b]] = e.matrix() @ t[[a, b]]
            elif isinstance(e, PhaseShifter):
                t[e.mode] *= np.exp(1j * e.phase)
            elif isinstance(e, Permutation):
                new = t.copy()
                for i, out in enumerate(e.perm):
                    new[e.modes[out]] = t[e.modes[i]]
                t = new
            elif isinstance(e, Loss):
                t[e.mode] *= math.sqrt(1.0 - e.loss)
            else:
                idx = list(e.modes)
                t[idx] = e.matrix @ t[idx]
        return t

    def is_lossless(self, tol: float = 1e-12) -> bool:
        t = self.matrix()
        return bool(np.allclose(t.conj().T @ t, np.eye(self.n_modes), atol=tol))

    def loss_channels(self) -> int:
        return sum(isinstance(e, Loss) and e.loss > 0 for e in self.elements)

    def copy(self) -> "Circuit":
        return Circuit(self.n_modes, list(self.elements))

    def to_dict(self) -> dict[str, Any]:
        return {"n_modes": self.n_modes, "elements": [_element_to_dict(e) for e in self.elements]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Circuit":
        try:
            c = cls(int(d["n_modes"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise FockError(f"invalid circuit: {exc}") from exc
        for raw in d.get("elements", []):
            c.add(_element_from_dict(raw))
        return c


def simplify_losses(circuit: Circuit) -> Circuit:
    """Merge per-mode loss channels met along a fixed path (across phase shifters and
    permutations); never commutes loss through mixing elements. Transfer matrix is unchanged."""
    out = Circuit(circuit.n_modes)
    pending = [1.0] * circuit.n_modes

    def flush(modes) -> None:
        for m in modes:
            if pending[m] < 1.0:
                out.add(Loss(m, 1.0 - pending[m]))
            pending[m] = 1.0

    for e in circuit.elements:
        if isinstance(e, Loss):
            pending[e.mode] *= 1.0 - e.loss
        elif isinstance(e, PhaseShifter):
            out.add(e)
        elif isinstance(e, Permutation):
            moved = list(pending)
            for i, o in enumerate(e.perm):
                moved[e.modes[o]] = pending[e.modes[i]]
            pending[:] = moved
            out.add(e)
        else:
            flush(e.modes)
            out.add(e)
    flush(range(circuit.n_modes))
    return out
