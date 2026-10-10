from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from sequoia.errors import Issue, UnknownComponentError
from sequoia.light.detection import DetectorModel, TaggerModel
from sequoia.light.pairs import PairEmission
from sequoia.light.state import Beam, LinkLight, SourceLight

CATEGORIES = (
    "sources",
    "nonlinear",
    "freespace",
    "polarization",
    "filters",
    "fiber",
    "detectors",
    "electronics",
    "probes",
)


class PortKind(StrEnum):
    FREE_SPACE = "free_space"
    FIBER = "fiber"
    ELECTRICAL = "electrical"


class PortDirection(StrEnum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True)
class PortSpec:
    name: str
    kind: PortKind
    direction: PortDirection
    required: bool = False


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class Context:
    issues: list[Issue] = field(default_factory=list)
    subject: str = ""
    instance_id: str = ""
    min_power_for_warnings: float = 1e-9
    sources: dict[str, SourceLight] = field(default_factory=dict)

    def beam_power(self, beam: Beam) -> float | None:
        """Power (W) carried by ``beam``, or None when its source did not register itself."""
        src = self.sources.get(beam.source)
        if src is None:
            return None
        weights = src.power * np.asarray(src.spectrum.weights, dtype=float)
        coherency = np.asarray(src.coherency, dtype=complex)
        out = beam.jones @ coherency @ np.conj(np.swapaxes(beam.jones, -1, -2))
        return float(np.sum(weights * np.real(np.trace(out, axis1=-2, axis2=-1))))

    def significant(self, beam: Beam) -> bool:
        power = self.beam_power(beam)
        return power is None or power >= self.min_power_for_warnings

    def warn(self, code: str, message: str) -> None:
        self.issues.append(Issue("warning", code, self.subject, message))


@dataclass
class Emission:
    source: SourceLight
    beams: list[tuple[str, Beam]]


class Component:
    category: ClassVar[str]
    type_name: ClassVar[str]
    Params: ClassVar[type[Params]] = Params
    Settings: ClassVar[type[Settings]] = Settings

    def __init__(self, params: Params, settings: Settings | None = None) -> None:
        self.params = params
        self.settings = settings if settings is not None else self.Settings()

    def ports(self) -> tuple[PortSpec, ...]:
        return ()

    def emit(self, ctx: Context) -> Emission | None:
        return None

    def propagate(self, port: str, beam: Beam, ctx: Context) -> list[tuple[str, Beam]]:
        return []

    def detect(self, light: dict[str, LinkLight], ctx: Context) -> dict[str, float]:
        return {}

    def emit_pairs(self, lights: dict[str, LinkLight], ctx: Context) -> PairEmission | None:
        """Called after the classical pass with the light arriving at each input port."""
        return None

    def detector_model(self) -> DetectorModel | None:
        return None

    def tagger_model(self) -> TaggerModel | None:
        return None

    def measure(self, light: LinkLight) -> dict:
        return {}


_REGISTRY: dict[str, type[Component]] = {}


def register(category: str, name: str):
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; choose from {', '.join(CATEGORIES)}")

    def decorator(cls: type[Component]) -> type[Component]:
        if name in _REGISTRY:
            raise ValueError(f"component type {name!r} is already registered")
        cls.category = category
        cls.type_name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_component(name: str) -> type[Component]:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise UnknownComponentError(
            f"unknown component type {name!r} (registered: {known})"
        ) from None


def registered_types(category: str | None = None) -> dict[str, type[Component]]:
    return {
        name: cls
        for name, cls in sorted(_REGISTRY.items())
        if category is None or cls.category == category
    }
