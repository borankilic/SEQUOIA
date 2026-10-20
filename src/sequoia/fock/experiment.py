import json
import os
import tempfile
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sequoia.fock.analysis import (
    Conditional,
    ExpectedCounts,
    Samples,
    check_counts,
    condition,
    expected_counts,
    format_state,
    sample,
)
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
    output_distribution_mixture,
)
from sequoia.fock.inputs import InputMixture

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
    input_mixture: InputMixture | None = None
    rate_hz: float | None = None
    window_s: float | None = None
    duration_s: float | None = None
    acquisition: str = "fixed_windows"

    def photons(self) -> list[tuple[int, Hashable]]:
        n = self.circuit.n_modes
        if self.input_mixture is not None:
            return []
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
        if self.input_mixture is not None:
            d["input_mixture"] = self.input_mixture.to_dict()
        counts = {
            k: v
            for k, v in (
                ("rate_hz", self.rate_hz),
                ("window_s", self.window_s),
                ("duration_s", self.duration_s),
            )
            if v is not None
        }
        if counts and self.acquisition != "fixed_windows":
            counts["acquisition"] = self.acquisition
        if counts:
            d["counts"] = counts
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
            mix_raw = raw.get("input_mixture")
            mixture = InputMixture.from_dict(mix_raw) if mix_raw else None
            cnt = raw.get("counts") or {}
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
                input_mixture=mixture,
                rate_hz=_opt_float(cnt.get("rate_hz")),
                window_s=_opt_float(cnt.get("window_s")),
                duration_s=_opt_float(cnt.get("duration_s")),
                acquisition=str(cnt.get("acquisition", "fixed_windows")),
            )
            check_counts(exp.rate_hz, exp.window_s, exp.duration_s)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise FockError(f"invalid experiment document: {exc}") from exc
        exp.photons()
        return exp


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


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
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=path.parent)
        os.close(fd)
        try:
            Path(tmp).write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
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
    counts: ExpectedCounts | None = None

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
            if self.counts and self.counts.per_second is not None:
                row["per_second"] = self.counts.per_second.get(s, 0.0)
                if self.counts.total is not None:
                    row["expected_counts"] = self.counts.total.get(s, 0.0)
                    row["sigma"] = self.counts.sigma.get(s, 0.0)
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
            if self.samples.unresolved_probability:
                d["unresolved_shots"] = self.samples.unresolved
                d["unresolved_probability"] = self.samples.unresolved_probability
        if self.raw.omitted_weight:
            d["omitted_weight"] = self.raw.omitted_weight
        if self.counts and self.counts.per_second is not None:
            d["expected_per_second"] = {
                format_state(s): v for s, v in self.counts.per_second.items()
            }
            if self.counts.total is not None:
                d["expected_counts"] = {format_state(s): v for s, v in self.counts.total.items()}
                d["duration_s"] = self.counts.duration_s
                d["n_windows"] = self.counts.n_windows
                d["sigma"] = {format_state(s): v for s, v in self.counts.sigma.items()}
                d["acquisition"] = self.counts.acquisition
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
    check_counts(exp.rate_hz, exp.window_s, exp.duration_s)
    layout = exp.layout()
    layout.check_efficiency(exp.circuit)
    photons = exp.photons()
    messages: list[str] = []
    backend = exp.backend
    if progress:
        progress(0, 4)
    mixture = exp.input_mixture
    if mixture is not None:
        if backend == "perceval":
            from sequoia.fock.perceval_bridge import perceval_mixture_distribution

            raw = perceval_mixture_distribution(exp.circuit, mixture)
        else:
            raw = output_distribution_mixture(
                exp.circuit, mixture, limits=exp.limits, should_cancel=should_cancel
            )
    elif backend == "perceval":
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
    detected.omitted_weight = raw.omitted_weight
    detected.notes = list(raw.notes)
    if progress:
        progress(2, 4)
    if abs(detected.mass + detected.omitted_weight - 1.0) > 1e-9:
        messages.append(f"total probability {detected.mass:.12g} differs from 1")
    if detected.omitted_weight > 1e-12:
        messages.append(
            f"input weight beyond the truncation, not simulated: {detected.omitted_weight:.3g}"
        )
    messages.extend(detected.notes)
    lost = raw.lost_mass(len(photons))
    if mixture is None and lost > 0.0 and exp.brightness == 1.0:
        messages.append(f"probability of losing at least one photon: {lost:.6g}")
    conditionals = {k: condition(detected, c) for k, c in exp.conditions.items()}
    samples = None
    if exp.shots > 0:
        samples = sample(detected, exp.shots, exp.seed, sequence=False)
    counts = None
    if exp.rate_hz is not None or exp.window_s is not None:
        duration = 1.0 if exp.duration_s is None else exp.duration_s
        counts = expected_counts(
            detected, exp.rate_hz, exp.window_s, duration, acquisition=exp.acquisition
        )
        detected.info["n_windows"] = counts.n_windows
    if progress:
        progress(4, 4)
    res = {
        "photons": raw.info.get("photons", len(photons)),
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
        counts,
    )
