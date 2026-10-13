from collections.abc import Sequence

import numpy as np

from sequoia.fock.circuit import (
    BeamSplitter,
    Block,
    Circuit,
    FockError,
    Loss,
    Permutation,
    PhaseShifter,
)
from sequoia.fock.engine import Distribution, Photon, mixed_distribution

INSTALL_HINT = (
    "Perceval is not installed; install the optional extra with: pip install 'sequoia[perceval]'"
)


def require_perceval():
    try:
        import perceval as pcvl
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return pcvl


def perceval_version() -> str | None:
    try:
        return require_perceval().__version__
    except ImportError:
        return None


def _span_unitary(pcvl, modes: Sequence[int], matrix: np.ndarray):
    lo, hi = min(modes), max(modes)
    full = np.eye(hi - lo + 1, dtype=complex)
    idx = [m - lo for m in modes]
    for i, a in enumerate(idx):
        for j, b in enumerate(idx):
            full[a, b] = matrix[i, j]
    return lo, pcvl.Unitary(pcvl.Matrix(full))


def to_perceval_processor(circuit: Circuit, backend: str = "SLOS"):
    """Element-by-element Perceval processor; each loss becomes a pcvl.LC (extra mode in SLOS)."""
    pcvl = require_perceval()
    proc = pcvl.Processor(backend, circuit.n_modes)
    for e in circuit.elements:
        if isinstance(e, BeamSplitter):
            a, b = e.modes
            maker = {"Rx": pcvl.BS.Rx, "Ry": pcvl.BS.Ry, "H": pcvl.BS.H}[e.convention]
            if b == a + 1:
                proc.add(a, maker(e.theta, e.phi_tl, e.phi_bl, e.phi_tr, e.phi_br))
            else:
                proc.add(*_span_unitary(pcvl, e.modes, e.matrix()))
        elif isinstance(e, PhaseShifter):
            proc.add(e.mode, pcvl.PS(e.phase))
        elif isinstance(e, Permutation):
            modes = list(e.modes)
            if modes == list(range(modes[0], modes[0] + len(modes))):
                proc.add(modes[0], pcvl.PERM(list(e.perm)))
            else:
                m = np.zeros((len(modes), len(modes)), dtype=complex)
                for i, o in enumerate(e.perm):
                    m[o, i] = 1.0
                proc.add(*_span_unitary(pcvl, modes, m))
        elif isinstance(e, Loss):
            if e.loss > 0.0:
                proc.add(e.mode, pcvl.LC(e.loss))
        elif isinstance(e, Block):
            u, s, vh = np.linalg.svd(e.matrix)
            if np.allclose(s, 1.0, atol=1e-12):
                proc.add(*_span_unitary(pcvl, e.modes, e.matrix))
                continue
            proc.add(*_span_unitary(pcvl, e.modes, vh))
            for m, sv in zip(e.modes, s, strict=True):
                if sv < 1.0:
                    proc.add(m, pcvl.LC(float(1.0 - sv**2)))
            proc.add(*_span_unitary(pcvl, e.modes, u))
        else:
            raise FockError(f"unsupported element {type(e).__name__}")
    return proc


def _noisy_state(pcvl, n_modes: int, photons: Sequence[Photon]):
    labels: dict = {}
    per_mode: list[list[int]] = [[] for _ in range(n_modes)]
    for m, tag in photons:
        per_mode[m].append(labels.setdefault(tag, len(labels)))
    parts = ["".join(f"{{{t}}}" for t in sorted(tags)) if tags else "0" for tags in per_mode]
    return pcvl.NoisyFockState("|" + ",".join(parts) + ">")


def perceval_distribution(
    circuit: Circuit | np.ndarray,
    photons: Sequence[Photon],
    backend: str = "SLOS",
) -> Distribution:
    pcvl = require_perceval()
    from perceval.algorithm import Sampler

    if not isinstance(circuit, Circuit):
        c = Circuit(np.asarray(circuit).shape[0])
        c.block(range(c.n_modes), circuit)
        circuit = c
    proc = to_perceval_processor(circuit, backend)
    proc.min_detected_photons_filter(0)
    proc.with_input(_noisy_state(pcvl, circuit.n_modes, photons))
    out = Sampler(proc).probs()
    if out is None:
        raise FockError("Perceval could not evaluate this circuit")
    probs: dict[tuple[int, ...], float] = {}
    for state, p in out["results"].items():
        key = tuple(int(x) for x in state)
        probs[key] = probs.get(key, 0.0) + float(p)
    return Distribution(
        dict(sorted(probs.items())),
        circuit.n_modes,
        {"backend": f"perceval-{backend}", "photons": len(photons)},
    )


def perceval_mixed_distribution(
    circuit: Circuit | np.ndarray,
    photons: Sequence[Photon],
    *,
    brightness: float = 1.0,
    indistinguishability: float = 1.0,
    backend: str = "SLOS",
) -> Distribution:
    t = circuit.matrix() if isinstance(circuit, Circuit) else circuit
    return mixed_distribution(
        t,
        photons,
        brightness=brightness,
        indistinguishability=indistinguishability,
        backend=lambda m, ph: perceval_distribution(
            circuit if isinstance(circuit, Circuit) else m, ph, backend
        ),
    )


def compare_with_perceval(
    circuit: Circuit, photons: Sequence[Photon], backend: str = "SLOS"
) -> float:
    """Maximum absolute probability difference between the native and the Perceval result."""
    from sequoia.fock.engine import output_distribution

    native = output_distribution(circuit, photons)
    other = perceval_distribution(circuit, photons, backend)
    keys = set(native.probs) | set(other.probs)
    return max((abs(native.probs.get(k, 0.0) - other.probs.get(k, 0.0)) for k in keys), default=0.0)
