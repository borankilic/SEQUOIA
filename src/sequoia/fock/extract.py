import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from sequoia.components.base import Component, Context, PortDirection, PortKind, get_component
from sequoia.engines.cache import check_cancel
from sequoia.engines.classical import LinkState, build_network, propagate_network
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
    FockCompileError,
    _guess_wavelength,
    _psd_sqrt,
    _reachable,
    compile_setup_network,
)
from sequoia.fock.experiment import DetectorSpec, Experiment
from sequoia.fock.setup_sources import _discover
from sequoia.library.store import Library
from sequoia.light.gaussian import GaussianMode, aperture_transmission
from sequoia.setup.model import Setup

if TYPE_CHECKING:
    from sequoia.fock.from_setup import SetupExperiment


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
    se: "SetupExperiment", wavelength: float | None = None, should_cancel=None
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
