from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from sequoia.data import Duration, Length, Scalar
from sequoia.measurement import Measurement

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
Id = Annotated[str, StringConstraints(pattern=rf"^{_NAME}$")]
Endpoint = Annotated[str, StringConstraints(pattern=rf"^{_NAME}\.{_NAME}$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _Placed(_Model):
    id: Id
    part: str
    settings: dict[str, Any] = {}
    overrides: dict[str, Any] = {}


class Instance(_Placed):
    position: Annotated[list[float], Field(min_length=2, max_length=2)] = [0.0, 0.0]
    rotation: float = 0.0
    flipped: bool = False

    @field_validator("rotation")
    @classmethod
    def _wrap_rotation(cls, value: float) -> float:
        return value % 360.0


class ProbeSpec(_Placed):
    link: Id


class _Link(_Model):
    id: Id
    src: Endpoint = Field(alias="from")
    dst: Endpoint = Field(alias="to")


def _non_negative(scalar: Scalar | None, what: str) -> None:
    if scalar is not None and not scalar.si >= 0:
        raise ValueError(f"{what} must not be negative")


class FreeSpaceLink(_Link):
    kind: Literal["free_space"]
    length: Length | None = None
    lock_to_canvas: bool = False
    medium: str = "air"

    @model_validator(mode="after")
    def _check(self) -> "FreeSpaceLink":
        if self.length is None and not self.lock_to_canvas:
            raise ValueError("a free-space link needs a 'length' or 'lock_to_canvas: true'")
        _non_negative(self.length, "length")
        return self


class FiberPolarization(_Model):
    kind: Literal["identity", "su2", "random"] = "identity"
    rotation_vector: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _check(self) -> "FiberPolarization":
        if self.kind == "su2" and self.rotation_vector is None:
            raise ValueError("kind 'su2' needs a 3-component 'rotation_vector' (radians)")
        if self.kind == "random" and self.seed is None:
            raise ValueError("kind 'random' needs a 'seed' so results are reproducible")
        if self.kind != "su2" and self.rotation_vector is not None:
            raise ValueError("'rotation_vector' only applies to kind 'su2'")
        if self.kind != "random" and self.seed is not None:
            raise ValueError("'seed' only applies to kind 'random'")
        return self


class FiberLink(_Link):
    kind: Literal["fiber"]
    fiber: str
    length: Length
    connector_loss_db: float = Field(default=0.0, ge=0.0)
    polarization: FiberPolarization = FiberPolarization()

    @model_validator(mode="after")
    def _check(self) -> "FiberLink":
        _non_negative(self.length, "length")
        return self


class CableLink(_Link):
    kind: Literal["cable"]
    delay: Duration = Scalar(value=0.0, unit="s")


Link = Annotated[FreeSpaceLink | FiberLink | CableLink, Field(discriminator="kind")]


class Setup(_Model):
    name: str = "untitled"
    description: str = ""
    grid_cell: Length = Scalar(value=25.0, unit="mm")
    instances: list[Instance] = []
    links: list[Link] = []
    probes: list[ProbeSpec] = []
    analyses: list[dict[str, Any]] = []
    measurement: Measurement | None = None

    @model_validator(mode="after")
    def _check_grid(self) -> "Setup":
        if not self.grid_cell.si > 0:
            raise ValueError("grid_cell must be positive")
        return self
