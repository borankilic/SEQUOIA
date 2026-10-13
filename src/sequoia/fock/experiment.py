import json
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sequoia.fock.analysis import Conditional, Samples, condition, format_state, sample
from sequoia.fock.circuit import Circuit, FockError
from sequoia.fock.detectors import (
    Detector,
    DetectorLayout,
    IdealPNR,
    detector_from_dict,
)
from sequoia.fock.engine import (
    Distribution,
    Limits,
    mixed_distribution,
)

BACKENDS = ("native", "perceval")
FORMAT_VERSION = 1


@dataclass
class DetectorSpec:
    modes: tuple[int, ...]
    model: Detector = field(default_factory=IdealPNR)


@dataclass
class Experiment:
    circuit: Circuit
    occupations: list[int]
    tags: list[list[Hashable]] | None = None
    brightness: float = 1.0
    indistinguishability: float = 1.0
    detectors: list[DetectorSpec] = field(default_factory=list)
    shots: int = 0
    seed: int | None = None
    backend: str = "native"
    conditions: dict[str, dict[int, Any]] = field(default_factory=dict)
    name: str = "photonic circuit experiment"
    description: str = ""
    mode_labels: list[str] | None = None
    limits: Limits = field(default_factory=Limits)

    def photons(self) -> list[tuple[int, Hashable]]:
        n = self.circuit.n_modes
        if len(self.occupations) != n:
            raise FockError(f"{len(self.occupations)} occupations for {n} modes")
        out = []
        for m, k in enumerate(self.occupations):
            if k < 0:
                raise FockError(f"negative occupation in mode {m}")
            tags = self.tags[m] if self.tags and m < len(self.tags) else []
            if tags and len(tags) != k:
                raise FockError(f"mode {m} has {k} photons but {len(tags)} tags")
            out.extend((m, tags[i] if tags else 0) for i in range(k))
        return out

    def layout(self) -> DetectorLayout:
        return DetectorLayout.build(
            self.circuit.n_modes, [(d.modes, d.model) for d in self.detectors]
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "format": "sequoia-fock",
            "version": FORMAT_VERSION,
            "name": self.name,
            "description": self.description,
            "circuit": self.circuit.to_dict(),
            "input": {
                "occupations": list(self.occupations),
                "tags": self.tags,
                "brightness": self.brightness,
                "indistinguishability": self.indistinguishability,
            },
            "detectors": [{"modes": list(s.modes), **s.model.to_dict()} for s in self.detectors],
            "shots": self.shots,
            "seed": self.seed,
            "backend": self.backend,
            "conditions": {
                k: {str(m): _spec_to_plain(v) for m, v in c.items()}
                for k, c in self.conditions.items()
            },
        }
        if self.mode_labels:
            d["mode_labels"] = list(self.mode_labels)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Experiment":
        try:
            circuit = Circuit.from_dict(raw["circuit"])
            inp = raw.get("input", {})
            n = circuit.n_modes
            occ = [int(x) for x in inp.get("occupations", [0] * n)]
            tags = inp.get("tags")
            dets = []
            for d in raw.get("detectors", []):
                dets.append(DetectorSpec(tuple(int(m) for m in d["modes"]), detector_from_dict(d)))
            conds = {
                str(k): {int(m): _spec_from_plain(v) for m, v in c.items()}
                for k, c in (raw.get("conditions") or {}).items()
            }
            backend = raw.get("backend", "native")
            if backend not in BACKENDS:
                raise FockError(f"backend must be one of {BACKENDS}, got {backend!r}")
            exp = cls(
                circuit,
                occ,
                tags,
                float(inp.get("brightness", 1.0)),
                float(inp.get("indistinguishability", 1.0)),
                dets,
                int(raw.get("shots", 0)),
                raw.get("seed"),
                backend,
                conds,
                raw.get("name", "photonic circuit experiment"),
                raw.get("description", ""),
                raw.get("mode_labels"),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise FockError(f"invalid experiment document: {exc}") from exc
        exp.photons()
        return exp


def _spec_to_plain(v: Any) -> Any:
    if callable(v):
        raise FockError("callable conditions cannot be saved")
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    return list(v) if isinstance(v, tuple) else v


def _spec_from_plain(v: Any) -> Any:
    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
        return (v[0], v[1])
    if isinstance(v, list):
        return set(v)
    return v


def save_experiment(exp: Experiment, path: str | Path) -> None:
    path = Path(path)
    data = exp.to_dict()
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        from sequoia.yamlio import dump_yaml

        dump_yaml(data, path)


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            from sequoia.yamlio import load_yaml

            raw = load_yaml(path)
    except (OSError, ValueError) as exc:
        raise FockError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FockError(f"{path} is not a photonic circuit experiment document")
    return Experiment.from_dict(raw)


@dataclass
class FockResult:
    experiment: Experiment
    raw: Distribution
    probabilities: dict[tuple[int, ...], float]
    samples: Samples | None
    conditionals: dict[str, Conditional]
    backend: str
    versions: dict[str, str | None]
    seed: int | None
    resources: dict[str, Any]
    messages: list[str] = field(default_factory=list)

    @property
    def mass(self) -> float:
        return float(sum(self.probabilities.values()))

    def table(self) -> list[dict[str, Any]]:
        counts = self.samples.counts if self.samples else {}
        rows = []
        for s, p in sorted(self.probabilities.items(), key=lambda kv: -kv[1]):
            row: dict[str, Any] = {"state": format_state(s), "probability": p}
            if self.samples:
                row["counts"] = counts.get(s, 0)
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.experiment.name,
            "backend": self.backend,
            "versions": self.versions,
            "seed": self.seed,
            "resources": self.resources,
            "mass": self.mass,
            "messages": self.messages,
            "probabilities": {format_state(s): p for s, p in self.probabilities.items()},
            "photon_number_mass": {str(k): v for k, v in self.raw.by_photon_number().items()},
            "conditionals": {
                k: {
                    "success_probability": c.success_probability,
                    "probabilities": {format_state(s): p for s, p in c.probs.items()},
                    "description": c.description,
                }
                for k, c in self.conditionals.items()
            },
        }
        if self.samples:
            d["counts"] = {format_state(s): c for s, c in self.samples.counts.items()}
            d["shots"] = self.samples.shots
        return d


def _versions() -> dict[str, str | None]:
    import numpy

    import sequoia
    from sequoia.fock.perceval_bridge import perceval_version

    return {
        "sequoia": getattr(sequoia, "__version__", "0.1.0"),
        "numpy": numpy.__version__,
        "perceval": perceval_version(),
    }


def run_experiment(
    source: "Experiment | dict | str | Path",
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FockResult:
    t0 = time.perf_counter()
    if isinstance(source, Experiment):
        exp = source
    elif isinstance(source, dict):
        exp = Experiment.from_dict(source)
    else:
        exp = load_experiment(source)
    layout = exp.layout()
    layout.check_efficiency(exp.circuit)
    photons = exp.photons()
    messages: list[str] = []
    backend = exp.backend
    if progress:
        progress(0, 4)
    if backend == "perceval":
        from sequoia.fock.perceval_bridge import perceval_mixed_distribution

        raw = perceval_mixed_distribution(
            exp.circuit,
            photons,
            brightness=exp.brightness,
            indistinguishability=exp.indistinguishability,
        )
    else:
        raw = mixed_distribution(
            exp.circuit,
            photons,
            brightness=exp.brightness,
            indistinguishability=exp.indistinguishability,
            limits=exp.limits,
            should_cancel=should_cancel,
        )
    if progress:
        progress(1, 4)
    detected = layout.apply(raw)
    if progress:
        progress(2, 4)
    if abs(detected.mass - 1.0) > 1e-9:
        messages.append(f"total probability {detected.mass:.12g} differs from 1")
    lost = raw.lost_mass(len(photons))
    if lost > 0.0 and exp.brightness == 1.0:
        messages.append(f"probability of losing at least one photon: {lost:.6g}")
    conditionals = {k: condition(detected, c) for k, c in exp.conditions.items()}
    samples = None
    if exp.shots > 0:
        samples = sample(detected, exp.shots, exp.seed)
    if progress:
        progress(4, 4)
    res = {
        "photons": len(photons),
        "modes": exp.circuit.n_modes,
        "permanents": raw.info.get("permanents", 0),
        "estimated_work": raw.info.get("work", 0),
        "loss_modes": raw.info.get("loss_modes", 0),
        "output_states": len(detected.probs),
        "seconds": time.perf_counter() - t0,
    }
    return FockResult(
        exp,
        raw,
        detected.probs,
        samples,
        conditionals,
        raw.info.get("backend", backend),
        _versions(),
        exp.seed,
        res,
        messages,
    )
