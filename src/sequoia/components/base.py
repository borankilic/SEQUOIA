from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from sequoia.errors import Issue, UnknownComponentError
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
