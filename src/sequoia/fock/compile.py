import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import Context, PortDirection, PortKind
from sequoia.engines.cache import check_cancel
from sequoia.engines.classical import build_network, propagate_network
from sequoia.fock.circuit import Circuit, FockError
from sequoia.fock.detectors import IdealPNR, Threshold
from sequoia.fock.experiment import DetectorSpec, Experiment
from sequoia.library.store import Library
from sequoia.light.gaussian import GaussianMode, aperture_transmission
from sequoia.light.state import Beam
from sequoia.setup.model import Setup

ALLOWED_CATEGORIES = {"freespace", "polarization", "filters", "fiber", "detectors"}
_S2 = math.sqrt(2.0)
BASES = {
    "HV": (("H", np.array([1, 0], complex)), ("V", np.array([0, 1], complex))),
    "DA": (("D", np.array([1, 1], complex) / _S2), ("A", np.array([1, -1], complex) / _S2)),
    "RL": (("R", np.array([1, -1j], complex) / _S2), ("L", np.array([1, 1j], complex) / _S2)),
}


class FockCompileError(FockError):
    pass


@dataclass(frozen=True)
class ModeLabel:
    instance: str
    port: str
    pol: str
    species: str = ""

    def __str__(self) -> str:
        base = f"{self.instance}.{self.port}:{self.pol}"
        return f"{base} ({self.species})" if self.species else base


@dataclass
class CompiledNetwork:
    circuit: Circuit
    input_modes: list[ModeLabel]
    output_modes: list[ModeLabel]
    detectors: list[DetectorSpec]
    detector_ids: list[str]
    wavelength: float
    transfer: np.ndarray
    warnings: list[str] = field(default_factory=list)
    efficiency_in_circuit: list[str] = field(default_factory=list)
    input_wavelengths: list[float] = field(default_factory=list)

    def input_index(self, instance: str, port: str, pol: str) -> int:
        label = ModeLabel(instance, port, pol)
        for i, m in enumerate(self.input_modes):
            if (m.instance, m.port, m.pol) == (label.instance, label.port, label.pol):
                return i
        raise ValueError(f"no input mode {label}")

    def experiment(
        self, photons: Sequence[tuple[str, int] | tuple[str, int, object]], **kw
    ) -> Experiment:
        """``photons``: (``"inst.port:H"``, count[, tag]) entries."""
        n = self.circuit.n_modes
        occ = [0] * n
        tags: list[list] = [[] for _ in range(n)]
        for entry in photons:
            label, count = entry[0], int(entry[1])
            tag = entry[2] if len(entry) > 2 else 0
            inst_port, _, pol = label.partition(":")
            inst, _, port = inst_port.partition(".")
            i = self.input_index(inst, port, pol)
            occ[i] += count
            tags[i].extend([tag] * count)
        return Experiment(
            self.circuit,
            occ,
            tags,
            detectors=list(self.detectors),
            mode_labels=[str(m) for m in self.input_modes] + [""] * (n - len(self.input_modes)),
            **kw,
        )


def _split(endpoint: str) -> tuple[str, str]:
    inst, _, port = endpoint.partition(".")
    if not inst or not port:
        raise FockCompileError(f"input port {endpoint!r} must be written 'instance.port'")
    return inst, port


def _guess_wavelength(net, warnings: list[str]) -> float:
    pump = None
    crystal = False
    for comp in net.components.values():
        wl = getattr(comp.params, "wavelength", None)
        if comp.category == "sources" and wl is not None and pump is None:
            pump = float(wl.si)
        if comp.category == "nonlinear":
            crystal = True
    if pump is None:
        raise FockCompileError("cannot infer the photon wavelength; pass wavelength=")
    if crystal:
        warnings.append(
            f"wavelength not given: assumed degenerate SPDC, 2 x pump = {2 * pump * 1e9:.1f} nm"
        )
        return 2.0 * pump
    warnings.append(f"wavelength not given: using the laser wavelength {pump * 1e9:.1f} nm")
    return pump


def _reachable(net, starts: set[str]) -> set[str]:
    nxt: dict[str, set[str]] = {}
    for link in net.setup.links:
        if link.kind == "cable":
            continue
        nxt.setdefault(link.src.split(".")[0], set()).add(link.dst.split(".")[0])
    seen = set(starts)
    queue = deque(starts)
    while queue:
        for n in nxt.get(queue.popleft(), ()):
            if n not in seen:
                seen.add(n)
                queue.append(n)
    return seen


def _psd_sqrt(op: np.ndarray) -> np.ndarray:
    h = (op + op.conj().T) / 2.0
    w, v = np.linalg.eigh(h)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.conj().T


def compile_setup_network(
    setup: Setup,
    library: Library,
    wavelength: float | None = None,
    *,
    inputs: Sequence[str],
    detectors: Sequence[str] | None = None,
    input_basis: str = "HV",
    seed_mode: GaussianMode | None = None,
    gate_window: float | None = None,
    should_cancel: Callable[[], bool] | None = None,
    wavelengths: Sequence[float] | None = None,
    bases: Sequence[tuple[tuple[str, str], np.ndarray] | None] | None = None,
    seed_modes: Sequence[GaussianMode | None] | None = None,
    species: Sequence[str] | None = None,
    detector_kind: "str | dict[str, str]" = "pnr",
    net=None,
) -> CompiledNetwork:
    """Transfer matrix from the polarization mode pairs of ``inputs`` ("instance.port") to the
    polarization mode pairs of the detectors, for photons of ``wavelength`` (m).

    Only passive linear optics reachable from the inputs is converted (free-space optics, wave
    plates, polarizers, filters, fibers, fiber splitters, single-photon detectors); sources and
    nonlinear crystals raise FockCompileError. Spatial-mode overlap between recombining paths is
    taken as perfect; fiber coupling and apertures are evaluated for ``seed_mode``. The detector
    efficiency is returned in the detector model, not in the circuit, unless it depends on
    polarization (then its square-root operator is part of the transfer, listed in
    ``efficiency_in_circuit``).

    ``wavelengths`` gives one wavelength per input (default ``wavelength``); ``bases`` one
    ``((name0, name1), 2x2 matrix)`` per input whose columns are the Jones vectors of the two
    input modes (default ``input_basis``); ``seed_modes`` the spatial mode of each input;
    ``species`` a label per input. A source output port ("OUT" direction) may be an input, and the
    same port may be listed several times with different wavelengths. ``detector_kind`` is
    "pnr" or "threshold", or a mapping from detector id to one of them.
    """
    if input_basis not in BASES:
        raise FockCompileError(f"input_basis must be one of {tuple(BASES)}")
    if not inputs:
        raise FockCompileError("give at least one input port ('instance.port')")
    kinds = [detector_kind] if isinstance(detector_kind, str) else list(detector_kind.values())
    if any(k not in ("pnr", "threshold") for k in kinds):
        raise FockCompileError("detector_kind must be 'pnr' or 'threshold'")
    if net is None:
        net = build_network(setup, library)
    warnings: list[str] = []
    specs = [_split(p) for p in inputs]
    for name, seq in (
        ("wavelengths", wavelengths),
        ("bases", bases),
        ("seed_modes", seed_modes),
        ("species", species),
    ):
        if seq is not None and len(seq) != len(specs):
            raise FockCompileError(f"{name} needs one entry per input ({len(specs)})")
    out_starts = set()
    for inst, port in specs:
        comp = net.components.get(inst)
        if comp is None:
            raise FockCompileError(f"unknown instance {inst!r}")
        names = {p.name: p for p in comp.ports()}
        if port not in names:
            raise FockCompileError(f"{inst!r} has no port {port!r} (ports: {sorted(names)})")
        if comp.detector_model() is not None and names[port].direction == PortDirection.IN:
            raise FockCompileError(f"input port {inst}.{port} is a detector")
        if names[port].direction == PortDirection.OUT:
            out_starts.add(inst)
    reach = _reachable(net, {i for i, _ in specs})
    unsupported = []
    for inst in sorted(reach - out_starts):
        comp = net.components[inst]
        if comp.category not in ALLOWED_CATEGORIES:
            unsupported.append(f"{inst} ({comp.type_name}, category {comp.category})")
    if unsupported:
        raise FockCompileError(
            "components not convertible to a photon-number circuit: "
            + "; ".join(unsupported)
            + ". Sources and nonlinear crystals are not converted; start the network "
            "downstream of them and give the photons as Fock inputs."
        )
    det_ids = (
        list(detectors)
        if detectors is not None
        else [i for i, c in net.components.items() if c.detector_model() is not None]
    )
    for d in det_ids:
        if d not in net.components or net.components[d].detector_model() is None:
            raise FockCompileError(f"{d!r} is not a single-photon detector")
    if wavelengths is not None:
        lams = [float(x) for x in wavelengths]
        lam = lams[0] if wavelength is None else float(wavelength)
    else:
        lam = float(wavelength) if wavelength is not None else _guess_wavelength(net, warnings)
        lams = [lam] * len(specs)
    default_basis = (
        tuple(n for n, _ in BASES[input_basis]),
        np.stack([v for _, v in BASES[input_basis]], axis=1),
    )
    n_in, n_out = 2 * len(specs), 2 * len(det_ids)
    r = np.zeros((n_out, n_in), dtype=complex)
    pol_names: list[str] = []
    multi_path = False
    for j, (inst, port) in enumerate(specs):
        check_cancel(should_cancel)
        owner = net.components[inst]
        direction = {p.name: p.direction for p in owner.ports()}[port]
        lam_j = lams[j]
        wl = np.array([lam_j])
        names_j, basis = bases[j] if bases is not None and bases[j] is not None else default_basis
        pol_names += list(names_j)
        mode = (
            seed_modes[j]
            if seed_modes is not None and seed_modes[j] is not None
            else seed_mode
            if seed_mode is not None
            else GaussianMode.from_waist(lam_j, 0.5e-3)
        )
        seed = Beam(
            f"fock:{inst}.{port}:{j}",
            wl,
            np.eye(2, dtype=complex)[None].copy(),
            0.0,
            0.0,
            mode,
            0.0,
            (),
        )
        ctx = Context(issues=[])

        def emit(inst_id, comp, ctx, seed=seed, inst=inst, port=port, direction=direction):
            if inst_id != inst:
                return []
            if direction == PortDirection.OUT:
                return [(port, seed)]
            return comp.apply_quality(seed, comp.propagate(port, seed, ctx), ctx)

        arrivals, _ = propagate_network(
            net, ctx, emit, coupling_note="", should_cancel=should_cancel
        )
        for issue in ctx.issues:
            msg = f"{issue.subject}: {issue.message}"
            if msg not in warnings:
                warnings.append(msg)
        for k, det_id in enumerate(det_ids):
            comp = net.components[det_id]
            block = np.zeros((2, 2), dtype=complex)
            count = 0
            for p in comp.ports():
                if p.direction != PortDirection.IN or p.kind == PortKind.ELECTRICAL:
                    continue
                for b in arrivals.get((det_id, p.name), []):
                    amp = b.jones[0] * np.exp(2j * np.pi * b.opl / lam_j)
                    radius_of = getattr(comp, "aperture_radius", None)
                    radius = radius_of() if radius_of is not None else None
                    if radius is not None:
                        amp = amp * math.sqrt(aperture_transmission(radius, b.mode))
                    block += amp
                    count += 1
            multi_path |= count > 1
            r[2 * k : 2 * k + 2, 2 * j : 2 * j + 2] = block @ basis
    if multi_path:
        warnings.append(
            "several paths recombine at a detector: perfect spatial-mode overlap is assumed"
        )
    models: list[DetectorSpec] = []
    in_circuit: list[str] = []
    out_labels: list[ModeLabel] = []
    unique = sorted(set(lams))
    for k, det_id in enumerate(det_ids):
        comp = net.components[det_id]
        model = comp.detector_model()
        ops = {lw: np.asarray(model.efficiency(np.array([lw])), dtype=complex)[0] for lw in unique}
        rows = slice(2 * k, 2 * k + 2)
        scalar = all(
            np.allclose(op, np.real(np.trace(op)) / 2.0 * np.eye(2), atol=1e-9)
            for op in ops.values()
        )
        if scalar:
            etas = np.array([float(np.real(np.trace(ops[lw])) / 2.0) for lw in lams])
            etas = np.clip(etas, 0.0, 1.0)
            ref = float(etas.max()) if etas.size else 0.0
            efficiency = ref
            if ref > 0.0:
                for j, e in enumerate(etas):
                    if abs(e - ref) > 1e-12:
                        r[rows, 2 * j : 2 * j + 2] *= math.sqrt(e / ref)
                        warnings.append(
                            f"detector {det_id}: efficiency differs between photon species; "
                            "the ratio is applied in the transfer matrix"
                        )
        else:
            for j, lw in enumerate(lams):
                cols = slice(2 * j, 2 * j + 2)
                r[rows, cols] = _psd_sqrt(ops[lw]) @ r[rows, cols]
            efficiency = 1.0
            in_circuit.append(det_id)
            warnings.append(
                f"detector {det_id}: polarization-dependent efficiency is applied as part of the "
                "transfer matrix"
            )
        dark = 0.0
        if gate_window is not None:
            dark = 1.0 - math.exp(-model.dark_count_rate * gate_window)
        kind = detector_kind if isinstance(detector_kind, str) else detector_kind.get(det_id, "pnr")
        cls = Threshold if kind == "threshold" else IdealPNR
        models.append(DetectorSpec((2 * k, 2 * k + 1), cls(efficiency=efficiency, dark=dark)))
        out_labels += [ModeLabel(det_id, "in", "H"), ModeLabel(det_id, "in", "V")]
    sv = np.linalg.norm(r, 2) if r.size else 0.0
    if sv > 1.0 + 1e-9:
        raise FockCompileError(
            f"compiled transfer is not contractive (largest singular value {sv:.6f}); "
            "an input port or component feeds more power than it receives"
        )
    n = max(n_in, n_out)
    t = np.zeros((n, n), dtype=complex)
    t[:n_out, :n_in] = r
    circuit = Circuit(n)
    circuit.block(range(n), t)
    inputs_labels = [
        ModeLabel(i, p, pol_names[2 * j + k], species[j] if species is not None else "")
        for j, (i, p) in enumerate(specs)
        for k in range(2)
    ]
    return CompiledNetwork(
        circuit,
        inputs_labels,
        out_labels,
        models,
        det_ids,
        lam,
        r,
        warnings,
        in_circuit,
        lams,
    )
