from pydantic import field_validator

from sequoia.components.base import (
    Component,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.data import Duration, Scalar
from sequoia.errors import SetupError
from sequoia.light.detection import TaggerModel


class TimeTaggerParams(Params):
    channels: int = 4
    jitter_fwhm: Duration = Scalar(value=0.0, unit="s")
    dead_time: Duration = Scalar(value=0.0, unit="s")

    @field_validator("channels")
    @classmethod
    def _channels(cls, v: int) -> int:
        if v < 1:
            raise ValueError("a time tagger needs at least one channel")
        return v


class TimeTaggerSettings(Settings):
    input_delays: dict[str, Duration] = {}


@register("electronics", "TimeTagger")
class TimeTagger(Component):
    Params = TimeTaggerParams
    Settings = TimeTaggerSettings

    def ports(self):
        return tuple(
            PortSpec(f"ch{i}", PortKind.ELECTRICAL, PortDirection.IN)
            for i in range(1, self.params.channels + 1)
        )

    def tagger_model(self):
        names = {p.name for p in self.ports()}
        unknown = sorted(set(self.settings.input_delays) - names)
        if unknown:
            raise SetupError(
                f"input_delays refers to unknown port(s) {', '.join(unknown)} "
                f"(ports: {', '.join(sorted(names))})"
            )
        return TaggerModel(
            jitter_fwhm=self.params.jitter_fwhm.si,
            dead_time=self.params.dead_time.si,
            input_delays={k: v.si for k, v in self.settings.input_delays.items()},
        )
