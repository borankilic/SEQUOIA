import numpy as np
import pytest

from sequoia.units import check_unit, from_si, to_si, to_wavelength, unit_dimension


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (405, "nm", 405e-9),
        (1, "um", 1e-6),
        (1, "µm", 1e-6),
        (2.5, "mW", 2.5e-3),
        (180, "deg", np.pi),
        (50, "%", 0.5),
        (3, "1", 3.0),
        (25, "degC", 298.15),
        (50, "ns", 50e-9),
        (10, "cps", 10.0),
    ],
)
def test_to_si(value, unit, expected):
    assert to_si(value, unit) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("unit", "factor"), [("cps", 1.0), ("kcps", 1e3), ("Mcps", 1e6)])
def test_counts_per_second_without_pint_warnings(unit, factor, caplog):
    assert to_si(2, unit) == pytest.approx(2 * factor)
    assert unit_dimension(unit) == "frequency"
    assert caplog.records == []


def test_to_si_array():
    result = to_si(np.array([1.0, 2.0]), "mm")
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [1e-3, 2e-3])


@pytest.mark.parametrize("unit", ["nm", "%", "degC", "THz", "1", "mW"])
def test_from_si_inverts_to_si(unit):
    assert from_si(to_si(7.5, unit), unit) == pytest.approx(7.5, rel=1e-12)


@pytest.mark.parametrize("unit", ["florps", "2 mm", "nm/"])
def test_unknown_unit(unit):
    with pytest.raises(ValueError, match="unknown unit"):
        check_unit(unit)


@pytest.mark.parametrize(
    ("unit", "dimension"),
    [
        ("nm", "length"),
        ("ns", "time"),
        ("THz", "frequency"),
        ("1/cm", "wavenumber"),
        ("eV", "energy"),
        ("%", "dimensionless"),
        ("A/W", "other"),
        ("degC", "temperature"),
        ("K", "temperature"),
        ("deg", "angle"),
        ("rad", "angle"),
        ("mrad", "angle"),
        ("arcmin", "angle"),
        ("mA", "current"),
        ("mV", "voltage"),
        ("cps", "frequency"),
        ("pm/V", "nonlinear_coefficient"),
        ("1", "dimensionless"),
    ],
)
def test_unit_dimension(unit, dimension):
    assert unit_dimension(unit) == dimension


def test_to_wavelength_matches_known_values():
    assert to_wavelength(to_si(1.0, "1/um"), "wavenumber") == pytest.approx(1e-6)
    assert to_wavelength(to_si(299.792458, "THz"), "frequency") == pytest.approx(1e-6)
    assert to_wavelength(to_si(1.239841984, "eV"), "energy") == pytest.approx(1e-6, rel=1e-6)
    with pytest.raises(ValueError):
        to_wavelength(1.0, "time")
