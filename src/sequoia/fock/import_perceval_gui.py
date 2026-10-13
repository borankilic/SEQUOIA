import json
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from sequoia.fock.circuit import BeamSplitter, Circuit, FockError, Permutation, db_to_loss
from sequoia.fock.detectors import IdealPNR, PseudoPNR, Threshold
from sequoia.fock.experiment import DetectorSpec, Experiment

SUPPORTED_VERSIONS = (1, 2)
_BS_CONVENTION = {
    "Beam Splitter H": "H",
    "Beam Splitter Rx": "Rx",
    "Beam Splitter Ry": "Ry",
    "Beam Splitter Adjustable": "Rx",
    "Beam Splitter Full": "Rx",
}
_DETECTORS = ("Threshold", "PNR", "PPNR", "BSLayeredPPNR")
_MZI = ("MZI First", "MZI Last", "MZI Last (top)", "MZI Symmetric")


class FockImportError(FockError):
    pass


@dataclass
class ImportedCircuit:
    experiment: Experiment
    noise: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    n_components: int = 0

    @property
    def circuit(self) -> Circuit:
        return self.experiment.circuit


def _num(boxes: dict[str, Any], key: str, where: str, default: float | None = None) -> float:
    if key not in boxes:
        if default is not None:
            return default
        raise FockImportError(f"{where}: missing field '{key}'")
    text = str(boxes[key]).strip()
    try:
        value = float(text)
    except ValueError:
        raise FockImportError(f"{where}: '{key}' = {text!r} is not a number") from None
    if not math.isfinite(value):
        raise FockImportError(f"{where}: '{key}' is not finite")
    return value


def _mode(boxes: dict[str, Any], key: str, n: int, where: str) -> int:
    v = _num(boxes, key, where)
    if v != int(v) or not 0 <= int(v) < n:
        raise FockImportError(f"{where}: '{key}' = {boxes[key]!r} is not a mode in 0..{n - 1}")
    return int(v)


class _Builder:
    def __init__(self, n: int, db_per_cm: float) -> None:
        self.c = Circuit(n)
        self.db_per_cm = db_per_cm

    def path(self, modes, length: float) -> None:
        loss = db_to_loss(self.db_per_cm * length)
        for m in modes:
            self.c.loss(m, loss)

    def db(self, modes, db: float) -> None:
        for m in modes:
            self.c.loss(m, db_to_loss(db))

    def ps(self, mode: int, phase: float, db: float) -> None:
        self.c.phase_shifter(mode, phase)
        self.db([mode], db)

    def bs(self, top: int, db: float) -> None:
        self.c.beam_splitter(top, top + 1)
        self.db([top, top + 1], db)


def import_perceval_gui(
    source: str | Path | dict[str, Any],
    occupations: list[int] | None = None,
    strict: bool = True,
) -> ImportedCircuit:
    """Native circuit from a Perceval GUI version 1/2 circuit file.

    Mirrors the GUI's circuit_builder: input mode losses first; per component the waveguide
    loss (db_per_cm * length, no phase) before the element and its active loss after it on both
    modes; a detector's path loss precedes it and its efficiency is kept in the detector
    response. Unknown components (or, with ``strict=False``, anything not representable) are
    collected in ``unsupported``; ``strict`` raises instead of skipping them.
    """
    if isinstance(source, dict):
        raw = source
    else:
        try:
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FockImportError(f"cannot read {source}: {exc}") from exc
    try:
        version = int(raw.get("version", 1))
        mode_losses = [str(x) for x in raw["mode_losses"]]
        comps = raw["components"]
        db_per_cm = float(raw["db_per_cm"])
    except (KeyError, TypeError, ValueError, AttributeError):
        raise FockImportError("not a Perceval GUI circuit file (version 1 or 2)") from None
    if version not in SUPPORTED_VERSIONS:
        raise FockImportError(f"unsupported Perceval GUI file version {version}")
    n = len(mode_losses)
    if n == 0:
        raise FockImportError("the circuit has no modes")
    out = ImportedCircuit(
        Experiment(Circuit(n), [0] * n),
        {"brightness": 1.0, "g2": 0.0, "indistinguishability": 1.0},
    )
    for key in out.noise:
        if key in (raw.get("noise") or {}):
            out.noise[key] = float(raw["noise"][key])
    b = _Builder(n, db_per_cm)
    for m, text in enumerate(mode_losses):
        out_loss = _num({"v": text}, "v", f"mode_losses[{m}]")
        b.c.loss(m, out_loss)
    detectors: dict[int, DetectorSpec] = {}
    for idx, comp in enumerate(comps):
        name = comp.get("name")
        boxes = comp.get("boxes", {})
        where = f"component {idx + 1} ({name})"
        out.n_components += 1
        if name == "Phase shifter":
            m = _mode(boxes, "mode", n, where)
            b.path([m], _num(boxes, "length", where))
            b.ps(m, _num(boxes, "phase", where), _num(boxes, "active_db", where))
        elif name == "Crossing" or name in _BS_CONVENTION:
            m1, m2 = _mode(boxes, "mode1", n, where), _mode(boxes, "mode2", n, where)
            b.path([m1], _num(boxes, "length1", where))
            b.path([m2], _num(boxes, "length2", where))
            top = min(m1, m2)
            if abs(m1 - m2) != 1:
                raise FockImportError(f"{where}: modes {m1} and {m2} are not consecutive")
            if name == "Crossing":
                b.c.add(Permutation((top, top + 1), (1, 0)))
            else:
                if m2 != m1 + 1:
                    raise FockImportError(f"{where}: mode 2 must be mode 1 + 1")
                theta = (
                    math.pi / 2
                    if name in ("Beam Splitter H", "Beam Splitter Rx", "Beam Splitter Ry")
                    else _num(boxes, "theta", where)
                )
                phases = {}
                if name == "Beam Splitter Full":
                    phases = {
                        k: _num(boxes, k, where) for k in ("phi_tl", "phi_tr", "phi_bl", "phi_br")
                    }
                b.c.add(BeamSplitter((m1, m2), theta, convention=_BS_CONVENTION[name], **phases))
            b.db([m1, m2], _num(boxes, "active_db", where))
        elif name in _MZI:
            top = _mode(boxes, "mode1", n, where)
            bottom = _mode(boxes, "mode2", n, where)
            if bottom != top + 1:
                raise FockImportError(f"{where}: mode 2 must be mode 1 + 1")
            both = [top, bottom]
            g = partial(_num, boxes, where=where)
            if name == "MZI First":
                b.path(both, g("to_ps1"))
                b.ps(top, g("phi_a"), g("ps_db"))
                b.path(both, g("ps1_to_bs1"))
                b.bs(top, g("bs_db"))
                b.path(both, g("bs1_to_ps2"))
                b.ps(top, g("phi_b"), g("ps_db"))
                b.path(both, g("ps2_to_bs2"))
                b.bs(top, g("bs_db"))
            elif name in ("MZI Last", "MZI Last (top)"):
                ps_mode = top if name == "MZI Last (top)" else bottom
                b.path(both, g("to_bs1"))
                b.bs(top, g("bs_db"))
                b.ps(ps_mode, g("phi_a"), g("ps_db"))
                b.path(both, g("ps1_to_bs2"))
                b.bs(top, g("bs_db"))
                b.path(both, g("bs2_to_ps2"))
                b.ps(ps_mode, g("phi_b"), g("ps_db"))
            else:
                b.path(both, g("to_bs1"))
                b.bs(top, g("bs_db"))
                b.path(both, g("bs1_to_ps"))
                b.ps(top, g("phi_a"), g("ps_db"))
                b.ps(bottom, g("phi_b"), g("ps_db"))
                b.path(both, g("ps_to_bs2"))
                b.bs(top, g("bs_db"))
        elif name in _DETECTORS:
            m = _mode(boxes, "mode", n, where)
            b.path([m], _num(boxes, "length", where))
            eff = _num(boxes, "efficiency", where)
            if not 0.0 <= eff <= 1.0:
                raise FockImportError(f"{where}: efficiency {eff} outside [0, 1]")
            if m in detectors:
                raise FockImportError(f"{where}: mode {m} already has a detector")
            if name == "Threshold":
                model = Threshold(efficiency=eff)
            elif name == "PNR":
                model = IdealPNR(efficiency=eff)
            elif name == "PPNR":
                model = PseudoPNR(wires=int(_num(boxes, "wires", where)), efficiency=eff)
            else:
                model = PseudoPNR.layered(
                    int(_num(boxes, "layers", where)),
                    _num(boxes, "reflectivity", where),
                    efficiency=eff,
                )
            detectors[m] = DetectorSpec((m,), model)
        else:
            out.unsupported.append(f"{where}: unknown component")
    if out.noise["g2"] > 0.0:
        out.unsupported.append(
            f"noise: g2 = {out.noise['g2']} (multi-photon component) is not modelled and is ignored"
        )
    if out.unsupported and strict:
        raise FockImportError("unsupported content:\n" + "\n".join(out.unsupported))
    out.warnings.extend(out.unsupported)
    exp = out.experiment
    exp.circuit = b.c
    exp.detectors = [detectors[m] for m in sorted(detectors)]
    exp.brightness = out.noise["brightness"]
    exp.indistinguishability = out.noise["indistinguishability"]
    exp.occupations = list(occupations) if occupations is not None else [0] * n
    exp.name = Path(source).stem if not isinstance(source, dict) else "perceval gui circuit"
    return out
