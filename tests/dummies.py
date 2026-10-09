import datetime
from typing import Literal

from sequoia.components import base
from sequoia.components.base import (
    Component,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.data import Curve, FileRef, Provenance, Scalar, Surface

IN, OUT = PortDirection.IN, PortDirection.OUT
FREE, FIBER, ELEC = PortKind.FREE_SPACE, PortKind.FIBER, PortKind.ELECTRICAL


class LaserParams(Params):
    wavelength: Scalar
    power: Scalar = Scalar(value=1.0, unit="mW")


class AngleSettings(Settings):
    angle: Scalar = Scalar(value=0.0, unit="deg")


class FilterParams(Params):
    transmission: Curve
    optical_density: Scalar | None = None


class CrystalParams(Params):
    length: Scalar
    brightness: Surface
    datasheet: FileRef | None = None


class FiberParams(Params):
    mode_field_diameter: Scalar
    attenuation: Curve | None = None


class CouplerSettings(Settings):
    mode: Literal["to_fiber", "to_free"] = "to_fiber"


class DetectorParams(Params):
    efficiency: Curve
    dark_count_rate: Scalar = Scalar(value=100.0, unit="Hz")
    dead_time: Scalar = Scalar(value=50.0, unit="ns")


class EmptyParams(Params):
    pass


class DummyLaser(Component):
    Params = LaserParams

    def ports(self):
        return (PortSpec("out", FREE, OUT),)


class DummyMirror(Component):
    Params = EmptyParams
    Settings = AngleSettings

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))


class DummyBeamSplitter(Component):
    Params = EmptyParams

    def ports(self):
        return (
            PortSpec("in_a", FREE, IN),
            PortSpec("in_b", FREE, IN),
            PortSpec("out_t", FREE, OUT),
            PortSpec("out_r", FREE, OUT),
        )


class DummyFilter(Component):
    Params = FilterParams

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))


class DummyCrystal(Component):
    Params = CrystalParams

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))


class DummyCoupler(Component):
    Params = EmptyParams
    Settings = CouplerSettings

    def ports(self):
        if self.settings.mode == "to_fiber":
            return (PortSpec("in", FREE, IN), PortSpec("out", FIBER, OUT))
        return (PortSpec("in", FIBER, IN), PortSpec("out", FREE, OUT))


class DummyFiber(Component):
    Params = FiberParams


class DummyDetector(Component):
    Params = DetectorParams

    def ports(self):
        return (PortSpec("in", FIBER, IN, required=True), PortSpec("out", ELEC, OUT, required=True))


class DummyTagger(Component):
    Params = EmptyParams

    def ports(self):
        return tuple(PortSpec(f"ch{i}", ELEC, IN) for i in range(1, 5))


class DummyPowerMeter(Component):
    Params = EmptyParams


SPECS = (
    ("sources", "DummyLaser", DummyLaser),
    ("freespace", "DummyMirror", DummyMirror),
    ("freespace", "DummyBeamSplitter", DummyBeamSplitter),
    ("filters", "DummyFilter", DummyFilter),
    ("nonlinear", "DummyCrystal", DummyCrystal),
    ("fiber", "DummyCoupler", DummyCoupler),
    ("fiber", "OpticalFiber", DummyFiber),
    ("detectors", "DummyDetector", DummyDetector),
    ("electronics", "DummyTagger", DummyTagger),
    ("probes", "DummyPowerMeter", DummyPowerMeter),
)


def register_all() -> None:
    for category, name, cls in SPECS:
        register(category, name)(cls)


def reset_registry(monkeypatch) -> None:
    monkeypatch.setattr(base, "_REGISTRY", {})


def make_curve(**kwargs) -> Curve:
    defaults = dict(
        x=[400.0, 500.0, 600.0],
        y=[0.1, 0.9, 0.5],
        x_label="wavelength",
        x_unit="nm",
        y_label="transmission",
    )
    return Curve(**{**defaults, **kwargs})


def make_parts(library) -> None:
    from sequoia.library.store import Part

    measured = Provenance(source="measured", date=datetime.date(2026, 3, 1), by="Boran")
    empty = EmptyParams()
    parts = [
        Part("sources/laser", "DummyLaser", LaserParams(wavelength=Scalar(value=405, unit="nm"))),
        Part("freespace/mirror", "DummyMirror", empty),
        Part("freespace/bs", "DummyBeamSplitter", empty),
        Part(
            "filters/bp",
            "DummyFilter",
            FilterParams(transmission=make_curve(provenance=measured)),
        ),
        Part("fiber/coupler", "DummyCoupler", empty),
        Part(
            "fiber/smf",
            "OpticalFiber",
            FiberParams(mode_field_diameter=Scalar(value=5.0, unit="um")),
        ),
        Part(
            "detectors/spad",
            "DummyDetector",
            DetectorParams(efficiency=make_curve(y_label="efficiency", y=[0.5, 0.6, 0.4])),
        ),
        Part("electronics/tagger", "DummyTagger", empty),
        Part("probes/powermeter", "DummyPowerMeter", empty),
    ]
    for part in parts:
        library.save_part(part)
