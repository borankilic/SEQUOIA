import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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


class OpticParams(Params):
    """Quality data shared by passive optics (catalogue 'specifications' tab)."""

    wavefront_error: float = Field(
        default=0.0,
        ge=0.0,
        description="RMS wavefront error in waves at 633 nm; degrades the beam quality (M^2)",
    )
    damage_threshold_cw: float | None = Field(
        default=None,
        gt=0.0,
        description="continuous-wave damage threshold in W/cm^2 (a warning is raised near it)",
    )
    damage_threshold_pulsed: float | None = Field(
        default=None,
        gt=0.0,
        description="pulsed damage threshold in J/cm^2 for the pulse length of the source",
    )


DAMAGE_CAUTION = 0.5


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

    def apply_quality(
        self, beam: Beam, outputs: list[tuple[str, Beam]], ctx: Context
    ) -> list[tuple[str, Beam]]:
        """Wavefront-error beam degradation and damage-threshold checks for passive optics."""
        p = self.params
        error = getattr(p, "wavefront_error", 0.0) or 0.0
        cw = getattr(p, "damage_threshold_cw", None)
        pulsed = getattr(p, "damage_threshold_pulsed", None)
        if cw is not None or pulsed is not None:
            power = ctx.beam_power(beam)
            src = ctx.sources.get(beam.source)
            if power is not None and power > 0 and ctx.significant(beam):
                area = math.pi * beam.mode.w_x * beam.mode.w_y / 2.0
                rep = getattr(src, "repetition_rate", None)
                if pulsed is not None and rep:
                    fluence = power / rep / area / 1e4
                    self._damage_warning(ctx, fluence, pulsed, "J/cm^2", "pulse fluence")
                elif cw is not None:
                    self._damage_warning(ctx, power / area / 1e4, cw, "W/cm^2", "peak intensity")
        if error <= 0.0:
            return outputs
        factor = math.sqrt(1.0 + (2.0 * math.pi * error) ** 2)
        return [(port, _degrade_beam(b, factor)) for port, b in outputs]

    @staticmethod
    def _damage_warning(ctx: Context, value: float, limit: float, unit: str, what: str) -> None:
        if value >= limit:
            ctx.warn(
                "damage_threshold_exceeded",
                f"{what} {value:.3g} {unit} exceeds the damage threshold of {limit:.3g} {unit}",
            )
        elif value >= DAMAGE_CAUTION * limit:
            ctx.warn(
                "damage_threshold_close",
                f"{what} {value:.3g} {unit} is {100 * value / limit:.0f}% of the damage "
                f"threshold of {limit:.3g} {unit}",
            )

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


def _degrade_beam(beam: Beam, factor: float) -> Beam:
    from dataclasses import replace

    return replace(beam, mode=beam.mode.degraded(factor))


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
