import math

import pytest
from optics import LASER, basic_library, fs, inst, probe, put, run
from pydantic import TypeAdapter, ValidationError

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import get_component
from sequoia.data import (
    Angle,
    Current,
    Dimensionless,
    Frequency,
    Rate,
    Scalar,
    Temperature,
    Voltage,
)
from sequoia.errors import SetupError
from sequoia.library.store import Library


def parse(annotation, value):
    return TypeAdapter(annotation).validate_python(value)


@pytest.mark.parametrize(
    ("value", "kelvin"),
    [("25 degC", 298.15), ("300 K", 300.0), ("-20 degC", 253.15), ("77 K", 77.0), ("0 K", 0.0)],
)
def test_temperature_accepts_temperature_units(value, kelvin):
    assert parse(Temperature, value).si == pytest.approx(kelvin)


def test_bare_temperature_is_an_error_with_a_hint():
    with pytest.raises(ValidationError, match=r"bare number.*25 degC"):
        parse(Temperature, 25)


@pytest.mark.parametrize("value", ["25 mW", "25 mm", "25 deg", "25 W"])
def test_temperature_rejects_other_dimensions(value):
    with pytest.raises(ValidationError, match="temperature"):
        parse(Temperature, value)


@pytest.mark.parametrize("value", ["-300 degC", "-1 K", "-500 degF"])
def test_temperature_below_absolute_zero_is_rejected(value):
    with pytest.raises(ValidationError, match="absolute zero"):
        parse(Temperature, value)


@pytest.mark.parametrize(
    ("value", "radians"),
    [
        ("45 deg", math.pi / 4),
        ("0.5 rad", 0.5),
        ("10 mrad", 0.01),
        ("30 arcmin", math.radians(0.5)),
        ("-12 deg", -math.radians(12)),
    ],
)
def test_angle_accepts_angle_units(value, radians):
    assert parse(Angle, value).si == pytest.approx(radians)


def test_bare_angle_is_rejected_with_a_hint():
    with pytest.raises(ValidationError, match=r"bare number.*45 deg"):
        parse(Angle, 45)
    with pytest.raises(ValidationError, match="bare number"):
        parse(Angle, "0.785")


def test_bare_zero_angle_is_unambiguous():
    assert parse(Angle, 0).si == 0.0


@pytest.mark.parametrize("value", ["45 mm", "45 degC", "3 mW", "1 Hz"])
def test_angle_rejects_other_dimensions(value):
    with pytest.raises(ValidationError, match="angle"):
        parse(Angle, value)


def test_frequency_rate_current_voltage():
    assert parse(Frequency, "80 MHz").si == pytest.approx(80e6)
    assert parse(Rate, "100 cps").si == pytest.approx(100.0)
    assert parse(Current, "5 uA").si == pytest.approx(5e-6)
    assert parse(Voltage, "250 mV").si == pytest.approx(0.25)
    for annotation, bad in (
        (Frequency, "5 ns"),
        (Rate, "5 mW"),
        (Current, "5 V"),
        (Voltage, "5 A"),
    ):
        with pytest.raises(ValidationError):
            parse(annotation, bad)
    for annotation in (Frequency, Rate, Current, Voltage):
        with pytest.raises(ValidationError, match="bare number"):
            parse(annotation, 5)


def test_dimensionless_accepts_ratios_only():
    assert parse(Dimensionless, 1000).si == 1000.0
    assert parse(Dimensionless, "0.5").si == 0.5
    assert parse(Dimensionless, "50 %").si == pytest.approx(0.5)
    for bad in ("1 mW", "3 deg", "3 rad"):
        with pytest.raises(ValidationError):
            parse(Dimensionless, bad)


def settings_of(type_name, **kw):
    return get_component(type_name).Settings(**kw)


def params_of(type_name, **kw):
    return get_component(type_name).Params(**kw)


@pytest.mark.parametrize(
    ("type_name", "field"),
    [
        ("WavePlate", "angle"),
        ("Polarizer", "angle"),
        ("PBS", "angle"),
        ("Mirror", "angle_of_incidence"),
        ("SpectralFilter", "angle_of_incidence"),
        ("Laser", "polarization_angle"),
        ("FiberPaddleController", "angle_1"),
        ("FiberPaddleController", "angle_3"),
    ],
)
def test_every_angle_setting_requires_an_angle_unit(type_name, field):
    assert settings_of(type_name, **{field: "30 deg"})
    with pytest.raises(ValidationError, match=field) as exc:
        settings_of(type_name, **{field: 30})
    assert "deg" in str(exc.value)
    with pytest.raises(ValidationError, match=field):
        settings_of(type_name, **{field: "30 mm"})


def test_laser_current_is_a_current():
    settings_of("Laser", current="30 mA")
    with pytest.raises(ValidationError, match="current"):
        settings_of("Laser", current=30)
    with pytest.raises(ValidationError, match="current"):
        settings_of("Laser", current="30 mW")


def test_crystal_temperature_settings_and_params():
    s = settings_of("SPDCCrystal", temperature="25 degC")
    assert s.temperature.si == pytest.approx(298.15)
    for bad in (25, "25 mW", "-300 degC", "25 mm"):
        with pytest.raises(ValidationError, match="temperature"):
            settings_of("SPDCCrystal", temperature=bad)
        with pytest.raises(ValidationError, match="design_temperature"):
            params_of("SPDCCrystal", length="10 mm", d_eff="1 pm/V", design_temperature=bad)


def test_crystal_theta_requires_an_angle_unit_but_accepts_auto():
    base = dict(length="10 mm", d_eff="1 pm/V")
    assert params_of("SPDCCrystal", theta="auto", **base).theta == "auto"
    assert params_of("SPDCCrystal", theta="90 deg", **base).theta.si == pytest.approx(math.pi / 2)
    with pytest.raises(ValidationError, match="theta"):
        params_of("SPDCCrystal", theta=90, **base)


def test_crystal_d_eff_needs_its_unit():
    with pytest.raises(ValidationError, match="d_eff"):
        params_of("SPDCCrystal", length="10 mm", d_eff=3.9)
    with pytest.raises(ValidationError, match="d_eff"):
        params_of("SPDCCrystal", length="10 mm", d_eff="3.9 mm")


def test_crystal_dnu_and_pair_source_rates():
    with pytest.raises(ValidationError, match="dnu"):
        params_of("SPDCCrystal", length="10 mm", d_eff="1 pm/V", dnu=5)
    ok = dict(center_wavelength="810 nm", bandwidth="1 nm")
    assert params_of("EntangledPairSource", pair_rate="1000 cps", **ok).pair_rate.si == 1000.0
    with pytest.raises(ValidationError, match="pair_rate"):
        params_of("EntangledPairSource", pair_rate=1000, **ok)
    with pytest.raises(ValidationError, match="phase"):
        params_of("EntangledPairSource", brightness=1e5, phase=0.5, **ok)
    assert params_of("EntangledPairSource", brightness=1e5, phase="0.5 rad", **ok)
    with pytest.raises(ValidationError, match="repetition_rate"):
        params_of("EntangledPairSource", brightness=1e5, repetition_rate="80 ns", **ok)


def test_detector_rates_times_and_angles():
    base = dict(efficiency="0.5")
    assert params_of("SPAD", dark_count_rate="100 cps", dead_time="50 ns", **base)
    with pytest.raises(ValidationError, match="dark_count_rate"):
        params_of("SPAD", dark_count_rate=100, **base)
    with pytest.raises(ValidationError, match="dark_count_rate"):
        params_of("SPAD", dark_count_rate="100 ns", **base)
    with pytest.raises(ValidationError, match="dead_time"):
        params_of("SPAD", dead_time=50, **base)
    with pytest.raises(ValidationError, match="polarization_axis"):
        params_of("SNSPD", polarization_axis=45, **base)
    assert params_of("SNSPD", polarization_axis="45 deg", **base)
    with pytest.raises(ValidationError, match="dark_current"):
        params_of("Photodiode", responsivity="0.5 A/W", dark_current="2 nW")


def test_bias_current_needs_a_current_unit():
    assert settings_of("SNSPD", bias_current="10 uA").bias_current.si == pytest.approx(1e-5)
    for bad in (10, "10 mW", "10 V"):
        with pytest.raises(ValidationError, match="bias_current"):
            settings_of("SNSPD", bias_current=bad)


def test_dimensionless_part_fields_keep_working():
    assert params_of("Polarizer", extinction_ratio=1000).extinction_ratio.si == 1000.0
    assert params_of("NDFilter", optical_density="2").optical_density.si == 2.0
    assert params_of("FiberBeamSplitter", split_ratio=0.4).split_ratio.si == 0.4
    assert params_of("Laser", wavelength="780 nm", power="1 mW", m_squared=1.1)
    with pytest.raises(ValidationError, match="extinction_ratio"):
        params_of("Polarizer", extinction_ratio="1000 mW")


def test_laser_linewidth_accepts_length_or_frequency_and_divergence_an_angle():
    base = dict(wavelength="780 nm", power="1 mW")
    assert params_of("Laser", linewidth="1 MHz", **base)
    assert params_of("Laser", linewidth="0.01 nm", **base)
    with pytest.raises(ValidationError, match="linewidth"):
        params_of("Laser", linewidth="1 mW", **base)
    assert params_of("Laser", divergence="1 mrad", **base)
    with pytest.raises(ValidationError, match="divergence"):
        params_of("Laser", divergence=0.001, **base)
    with pytest.raises(ValidationError, match="repetition_rate"):
        params_of("Laser", repetition_rate="80 ns", **base)


def test_setup_with_bare_angle_names_the_instance_and_field(tmp_path):
    lib = basic_library(Library(tmp_path / "lib"))
    instances = [inst("src", "laser"), inst("pol", "polarizer", {"angle": 45}), inst("end", "pd")]
    links = [fs("l0", "src.out", "pol.in"), fs("l1", "pol.out", "end.in")]
    with pytest.raises(SetupError, match=r"pol.*\n?.*angle.*45 deg"):
        run(lib, instances, links, [probe("p", "l1")])


def test_setup_with_temperature_in_milliwatts_is_rejected(tmp_path):
    lib = Library(tmp_path / "lib")
    put(lib, "laser", "Laser", **LASER)
    put(
        lib,
        "xtal",
        "SPDCCrystal",
        length="10 mm",
        d_eff="1 pm/V",
        poling_period="10 um",
        signal_wavelength="1560 nm",
    )
    instances = [inst("src", "laser"), inst("x", "xtal", {"temperature": "25 mW"})]
    with pytest.raises(SetupError, match="temperature"):
        run(lib, instances, [fs("l0", "src.out", "x.in")])


def test_yaml_shorthand_still_works():
    s = settings_of("WavePlate", angle="45 deg")
    assert s.angle.si == pytest.approx(math.pi / 4)
    assert isinstance(s.angle, Scalar)


@pytest.mark.parametrize("value", ["inf K", "-inf degC", "+inf K"])
def test_non_finite_temperatures_are_rejected(value):
    with pytest.raises(ValidationError, match="finite"):
        parse(Temperature, value)


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        (Angle, "inf deg"),
        (Voltage, "inf V"),
        (Current, "-inf mA"),
        (Frequency, "inf Hz"),
        (Dimensionless, "inf"),
    ],
)
def test_non_finite_physical_quantities_are_rejected(annotation, value):
    with pytest.raises(ValidationError, match="finite"):
        parse(annotation, value)


def test_nan_is_never_a_quantity():
    for value in ("nan K", float("nan")):
        with pytest.raises(ValidationError):
            parse(Temperature, value)
    with pytest.raises(ValidationError, match="NaN"):
        Scalar(value=float("nan"), unit="W")


def test_an_infinite_aperture_diameter_remains_valid():
    from sequoia.data import Length

    assert parse(Length, "inf m").si == float("inf")
    assert settings_of("Iris").diameter.si == float("inf")


@pytest.mark.parametrize("value", ["45 turn", "0.25 turns", "50 grad", "1 revolution"])
def test_angle_refuses_turns_and_gradians(value):
    with pytest.raises(ValidationError, match="not supported"):
        parse(Angle, value)


@pytest.mark.parametrize("value", ["45 deg", "45 °", "0.5 rad", "3 mrad", "30 arcmin", "5 arcsec"])
def test_angle_still_accepts_the_documented_units(value):
    assert parse(Angle, value).si != 0.0


def test_unicode_minus_is_normalised():
    assert parse(Angle, "−45 deg").si == pytest.approx(-math.pi / 4)
    assert parse(Temperature, "−20 degC").si == pytest.approx(253.15)
    assert parse(Scalar, "−1.5e-3 mm").value == -1.5e-3


def test_brightness_dimension_is_checked_and_bare_numbers_stay_per_mw():
    ok = dict(center_wavelength="810 nm", bandwidth="1 nm")
    assert params_of("EntangledPairSource", brightness=1e5, **ok).brightness == 1e5
    assert params_of("EntangledPairSource", brightness="1e5 Hz/mW", **ok).brightness == (
        pytest.approx(1e5)
    )
    assert params_of("EntangledPairSource", brightness="1e8 1/(s W)", **ok).brightness == (
        pytest.approx(1e5)
    )
    for bad in ("1e5 mm", "1e5 mW", "1e5 Hz", -1, "inf"):
        with pytest.raises(ValidationError, match="brightness"):
            params_of("EntangledPairSource", brightness=bad, **ok)
    with pytest.raises(ValidationError, match="brightness"):
        params_of("SPDCCrystal", length="10 mm", brightness=-5.0)


def test_responsivity_needs_amperes_per_watt():
    assert params_of("Photodiode", responsivity="0.5 A/W").responsivity.si == pytest.approx(0.5)
    assert params_of("Photodiode", responsivity="500 mA/W").responsivity.si == pytest.approx(0.5)
    for bad in (0.5, "0.5 W", "0.5 A", "0.5 mm"):
        with pytest.raises(ValidationError, match="responsivity"):
            params_of("Photodiode", responsivity=bad)
