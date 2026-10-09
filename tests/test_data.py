import datetime
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from sequoia.data import (
    Curve,
    Duration,
    ExtrapolationError,
    ExtrapolationWarning,
    Length,
    Provenance,
    Scalar,
    Surface,
    format_axis,
    parse_axis,
)


@pytest.mark.parametrize(
    ("text", "value", "unit", "si"),
    [
        ("405 nm", 405.0, "nm", 405e-9),
        ("405nm", 405.0, "nm", 405e-9),
        ("1.5e-3 W", 1.5e-3, "W", 1.5e-3),
        ("2 1/cm", 2.0, "1/cm", 200.0),
        ("0.5", 0.5, "1", 0.5),
    ],
)
def test_scalar_shorthand(text, value, unit, si):
    scalar = Scalar.model_validate(text)
    assert (scalar.value, scalar.unit) == (value, unit)
    assert scalar.si == pytest.approx(si)


def test_scalar_dict_and_number_forms():
    assert Scalar.model_validate({"value": 50, "unit": "Hz"}).si == 50.0
    assert Scalar.model_validate(3).unit == "1"


@pytest.mark.parametrize("bad", ["abc", "5 mm 3", "3 florps", True, ""])
def test_scalar_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        Scalar.model_validate(bad)


def test_scalar_serialization_is_compact_without_provenance():
    assert Scalar(value=100.0, unit="mm").model_dump() == "100 mm"
    assert Scalar(value=1e-9, unit="s").model_dump() == "1e-09 s"
    assert Scalar(value=0.25, unit="1").model_dump() == 0.25
    prov = Provenance(source="datasheet", date=datetime.date(2026, 3, 1))
    dumped = Scalar(value=3.0, unit="ns", provenance=prov).model_dump(exclude_none=True)
    assert dumped == {
        "value": 3.0,
        "unit": "ns",
        "provenance": {"source": "datasheet", "date": datetime.date(2026, 3, 1)},
    }


@pytest.mark.parametrize("value", [0.1, 1e-9, 123456789.123, 5.0, float("inf")])
def test_scalar_shorthand_roundtrips_exactly(value):
    scalar = Scalar(value=value, unit="m")
    assert Scalar.model_validate(scalar.model_dump()) == scalar


def test_dimension_checked_types():
    from pydantic import TypeAdapter

    assert TypeAdapter(Length).validate_python("3 mm").si == pytest.approx(3e-3)
    assert TypeAdapter(Duration).validate_python("3 ns").si == pytest.approx(3e-9)
    with pytest.raises(ValidationError, match="length quantity"):
        TypeAdapter(Length).validate_python("3 ns")


def test_provenance_rejects_unknown_source_and_fields():
    with pytest.raises(ValidationError):
        Provenance(source="guess")
    with pytest.raises(ValidationError):
        Provenance(source="measured", colour="red")


def make_curve(**kwargs):
    defaults = dict(
        x=[400.0, 500.0, 600.0],
        y=[0.1, 0.9, 0.5],
        x_label="wavelength",
        x_unit="nm",
        y_label="transmission",
    )
    return Curve(**{**defaults, **kwargs})


def test_curve_interpolates_in_si():
    curve = make_curve()
    np.testing.assert_allclose(curve.x_si, [400e-9, 500e-9, 600e-9])
    assert curve(450e-9) == pytest.approx(0.5)
    np.testing.assert_allclose(curve(np.array([400e-9, 550e-9])), [0.1, 0.7])


def test_curve_y_unit_conversion():
    curve = make_curve(y=[10.0, 90.0, 50.0], y_unit="%")
    assert curve(500e-9) == pytest.approx(0.9)


def test_curve_extrapolation_policies():
    with pytest.warns(ExtrapolationWarning, match="transmission"):
        assert make_curve()(1e-6) == pytest.approx(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert make_curve(extrapolation="clamp")(1e-6) == pytest.approx(0.5)
        assert make_curve(extrapolation="clamp")(1e-7) == pytest.approx(0.1)
        assert make_curve(extrapolation="zero")(1e-6) == 0.0
    with pytest.raises(ExtrapolationError):
        make_curve(extrapolation="error")(1e-6)


@pytest.mark.parametrize("policy", ["warn_clamp", "clamp", "zero", "error"])
def test_curve_edges_do_not_trip_on_unit_roundoff(policy):
    curve = make_curve(extrapolation=policy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert curve(400e-9) == pytest.approx(0.1)
        assert curve(600e-9) == pytest.approx(0.5)
        assert curve(np.nextafter(600e-9, 1.0)) == pytest.approx(0.5)


def test_curve_no_warning_inside_range():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_curve()(np.linspace(400e-9, 600e-9, 11))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(x=[400, 400, 600]), "strictly increasing"),
        (dict(x=[600, 500, 400]), "strictly increasing"),
        (dict(y=[0.1, 0.2]), "2 points|3 points|y has"),
        (dict(x=[400], y=[0.1]), "at least 2 points"),
        (dict(y=[0.1, float("nan"), 0.5]), "non-finite"),
        (dict(x_unit="florps"), "unknown unit"),
    ],
)
def test_curve_validation(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        make_curve(**kwargs)


def test_curve_equality_and_immutability():
    assert make_curve() == make_curve()
    assert make_curve() != make_curve(y=[0.1, 0.9, 0.6])
    with pytest.raises(ValidationError):
        make_curve().x_label = "other"


def test_surface_bilinear_and_policies():
    surface = Surface(
        x=[0.0, 1.0],
        y=[0.0, 10.0],
        z=[[0.0, 1.0], [2.0, 3.0]],
        x_unit="mW",
        y_unit="degC",
        z_label="brightness",
    )
    assert surface(0.5e-3, 5.0 + 273.15) == pytest.approx(1.5)
    with pytest.warns(ExtrapolationWarning, match="brightness"):
        assert surface(5e-3, 273.15) == pytest.approx(2.0)
    clamped = surface.model_copy(update={"extrapolation": "zero"})
    assert clamped(5e-3, 273.15) == 0.0


def test_surface_shape_is_checked():
    with pytest.raises(ValidationError, match="shape"):
        Surface(x=[0.0, 1.0], y=[0.0, 1.0], z=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])


def test_axis_parsing():
    assert parse_axis("wavelength (nm)") == ("wavelength", "nm")
    assert parse_axis("wavelength[nm]") == ("wavelength", "nm")
    assert parse_axis("efficiency ()") == ("efficiency", "1")
    assert parse_axis("rate (1/s)") == ("rate", "1/s")
    assert format_axis("wavelength", "nm") == "wavelength (nm)"
    with pytest.raises(ValueError, match="name \\(unit\\)"):
        parse_axis("wavelength nm")
