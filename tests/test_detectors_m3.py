import math
import warnings

import numpy as np
import pytest
from optics import fs, inst, put

from sequoia.components.base import PortDirection, PortKind, get_component
from sequoia.components.detectors import DetectorConfigWarning
from sequoia.data import Curve, Surface
from sequoia.engines.rates import simulate_rates
from sequoia.errors import SetupError
from sequoia.library.store import Library
from sequoia.setup.io import setup_from_dict
from sequoia.setup.validate import SetupValidationError, check_setup

WL = np.array([700e-9, 780e-9, 850e-9])


def make(type_name, settings=None, **params):
    cls = get_component(type_name)
    return cls(cls.Params.model_validate(params), cls.Settings.model_validate(settings or {}))


def operator(component, wavelengths=WL):
    return component.detector_model().efficiency(wavelengths)


def jones(angle_deg):
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


@pytest.mark.parametrize("type_name", ["SPAD", "SNSPD"])
def test_efficiency_is_hermitian_psd_with_max_eigenvalue_eta(type_name):
    extra = {"polarization_extinction": 7.0, "polarization_axis": "23 deg"}
    curve = Curve(x=[700.0, 900.0], y=[0.2, 0.8], x_unit="nm", x_label="wavelength")
    params = {"efficiency": curve, **(extra if type_name == "SNSPD" else {})}
    op = operator(make(type_name, **params))
    assert op.shape == (3, 2, 2)
    eta = 0.2 + 0.6 * (WL * 1e9 - 700) / 200
    for k in range(3):
        assert np.allclose(op[k], op[k].conj().T)
        w = np.linalg.eigvalsh(op[k])
        assert w.min() >= -1e-15
        assert w.max() == pytest.approx(eta[k])


def test_spad_is_polarization_insensitive():
    op = operator(make("SPAD", efficiency="0.6"))
    assert np.allclose(op, 0.6 * np.eye(2))


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 90.0, -20.0])
def test_snspd_response_follows_cos2_sin2_law(theta_deg):
    eta, ext, axis = 0.8, 5.0, theta_deg
    op = operator(
        make(
            "SNSPD",
            efficiency=eta,
            polarization_extinction=ext,
            polarization_axis=f"{axis} deg",
        )
    )[1]
    for psi in range(0, 181, 15):
        chi = jones(psi)
        response = (chi @ op @ chi).real
        delta = math.radians(psi - axis)
        expected = eta * (math.cos(delta) ** 2 + math.sin(delta) ** 2 / ext)
        assert response == pytest.approx(expected, abs=1e-12)
    chi = jones(axis)
    assert (chi @ op @ chi).real == pytest.approx(eta)
    chi = jones(axis + 90)
    assert (chi @ op @ chi).real == pytest.approx(eta / ext)


def test_snspd_circular_light_sees_mean_efficiency():
    op = operator(make("SNSPD", efficiency=0.9, polarization_extinction=4.0))[0]
    chi = np.array([1.0, 1.0j]) / math.sqrt(2)
    assert (chi.conj() @ op @ chi).real == pytest.approx(0.9 * (1 + 0.25) / 2)


def test_snspd_default_extinction_is_unpolarized():
    op = operator(make("SNSPD", efficiency=0.5))
    assert np.allclose(op, 0.5 * np.eye(2))


def test_snspd_rejects_extinction_below_one():
    with pytest.raises(ValueError):
        make("SNSPD", efficiency=0.5, polarization_extinction=0.5)


def test_efficiency_vs_bias_surface_and_bias_setting():
    surface = Surface(
        x=[700.0, 900.0],
        y=[8.0, 12.0],
        z=[[0.1, 0.5], [0.2, 0.9]],
        x_unit="nm",
        x_label="wavelength",
        y_unit="uA",
        y_label="bias",
    )
    det = make(
        "SNSPD",
        {"bias_current": "10 uA"},
        efficiency=0.3,
        efficiency_vs_bias=surface,
    )
    op = operator(det, np.array([800e-9]))
    assert op[0, 0, 0] == pytest.approx(np.mean([0.1, 0.5, 0.2, 0.9]))
    top = make("SNSPD", {"bias_current": "12 uA"}, efficiency=0.3, efficiency_vs_bias=surface)
    assert operator(top, np.array([900e-9]))[0, 0, 0] == pytest.approx(0.9)
    fallback = make("SNSPD", efficiency=0.3, efficiency_vs_bias=surface)
    with pytest.warns(DetectorConfigWarning):
        value = operator(fallback, np.array([800e-9]))[0, 0, 0]
    assert value == pytest.approx(0.3)


def test_bias_setting_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        make("SNSPD", {"bias_current": "3 nm"}, efficiency=0.3)


def test_curve_extrapolation_warns():
    curve = Curve(x=[750.0, 800.0], y=[0.4, 0.5], x_unit="nm", x_label="wavelength")
    det = make("SPAD", efficiency=curve)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op = operator(det, np.array([700e-9, 780e-9]))
    assert any("outside its data range" in str(w.message) for w in caught)
    assert op[0, 0, 0] == pytest.approx(0.4)
    assert op[1, 0, 0] == pytest.approx(0.46)


def test_detector_model_plumbing_in_si_units():
    det = make(
        "SNSPD",
        efficiency=0.8,
        dark_count_rate="2 kHz",
        dead_time="40 ns",
        jitter_fwhm="25 ps",
        jitter_tail_fraction=0.1,
        jitter_tail_tau="150 ps",
        afterpulse_probability=0.02,
        afterpulse_tau="1 us",
    )
    m = det.detector_model()
    assert m.kind == "SNSPD"
    assert m.dark_count_rate == pytest.approx(2e3)
    assert m.dead_time == pytest.approx(40e-9)
    assert m.jitter_fwhm == pytest.approx(25e-12)
    assert m.jitter_tail_fraction == pytest.approx(0.1)
    assert m.jitter_tail_tau == pytest.approx(150e-12)
    assert m.afterpulse_probability == pytest.approx(0.02)
    assert m.afterpulse_tau == pytest.approx(1e-6)


def test_defaults_are_ideal_noise_free():
    m = make("SPAD", efficiency=0.5).detector_model()
    assert (m.dark_count_rate, m.dead_time, m.jitter_fwhm, m.afterpulse_probability) == (0, 0, 0, 0)


def test_probability_params_are_bounded():
    with pytest.raises(ValueError):
        make("SPAD", efficiency=0.5, afterpulse_probability=1.5)


@pytest.mark.parametrize("type_name", ["SPAD", "SNSPD"])
def test_port_kinds_follow_input_setting(type_name):
    fiber = make(type_name, efficiency=0.5).ports()
    free = make(type_name, {"input": "free_space"}, efficiency=0.5).ports()
    by_name = {p.name: p for p in fiber}
    assert by_name["in"].kind == PortKind.FIBER
    assert by_name["in"].direction == PortDirection.IN and by_name["in"].required
    assert by_name["out"].kind == PortKind.ELECTRICAL
    assert by_name["out"].direction == PortDirection.OUT and by_name["out"].required
    assert {p.name: p.kind for p in free}["in"] == PortKind.FREE_SPACE


def test_photodiode_unchanged():
    pd = make("Photodiode", responsivity="0.5 A/W")
    assert pd.detector_model() is None
    assert [p.name for p in pd.ports()] == ["in"]


def test_time_tagger_ports_follow_channels():
    assert [p.name for p in make("TimeTagger").ports()] == ["ch1", "ch2", "ch3", "ch4"]
    eight = make("TimeTagger", channels=8).ports()
    assert [p.name for p in eight] == [f"ch{i}" for i in range(1, 9)]
    assert all(p.kind == PortKind.ELECTRICAL and p.direction == PortDirection.IN for p in eight)
    assert not any(p.required for p in eight)
    with pytest.raises(ValueError):
        make("TimeTagger", channels=0)


def test_time_tagger_model_plumbing():
    tagger = make(
        "TimeTagger",
        {"input_delays": {"ch2": "3 ns", "ch3": "-500 ps"}},
        channels=4,
        jitter_fwhm="40 ps",
        dead_time="2 ns",
    )
    m = tagger.tagger_model()
    assert m.jitter_fwhm == pytest.approx(40e-12)
    assert m.dead_time == pytest.approx(2e-9)
    assert m.input_delays == {"ch2": pytest.approx(3e-9), "ch3": pytest.approx(-0.5e-9)}
    assert make("TimeTagger").tagger_model().input_delays == {}


def test_time_tagger_rejects_delay_for_missing_channel():
    tagger = make("TimeTagger", {"input_delays": {"ch7": "1 ns"}}, channels=4)
    with pytest.raises(SetupError, match="ch7"):
        tagger.tagger_model()


def chain(library, with_cable=True, input_mode="free_space"):
    put(library, "laser", "Laser", wavelength="780 nm", power="1 mW", waist_radius="1 mm")
    put(library, "snspd", "SNSPD", efficiency=0.8)
    put(library, "tagger", "TimeTagger", channels=2)
    instances = [
        inst("src", "laser"),
        inst("det", "snspd", {"input": input_mode}),
        inst("tt", "tagger"),
    ]
    links = [fs("l0", "src.out", "det.in")]
    if with_cable:
        links.append(
            {"id": "c0", "kind": "cable", "from": "det.out", "to": "tt.ch1", "delay": "5 ns"}
        )
    return setup_from_dict({"instances": instances, "links": links})


def errors(issues):
    return [(i.code, i.subject) for i in issues if i.severity == "error"]


def test_validation_accepts_detector_cable_tagger_chain(tmp_path):
    library = Library(tmp_path / "lib")
    issues = check_setup(chain(library), library)
    assert errors(issues) == []


def test_validation_rejects_unconnected_detector_output(tmp_path):
    library = Library(tmp_path / "lib")
    issues = check_setup(chain(library, with_cable=False), library)
    assert ("missing_required", "port 'det.out'") in errors(issues)


def test_validation_rejects_free_space_into_fiber_detector(tmp_path):
    library = Library(tmp_path / "lib")
    issues = check_setup(chain(library, input_mode="fiber"), library)
    assert any(code == "kind_mismatch" for code, _ in errors(issues))


def test_validation_rejects_cable_into_nonexistent_tagger_channel(tmp_path):
    library = Library(tmp_path / "lib")
    setup = chain(library)
    raw = setup.model_dump(mode="python", by_alias=True)
    raw["links"][1]["to"] = "tt.ch3"
    issues = check_setup(setup_from_dict(raw), library)
    assert any(code == "bad_endpoint" for code, _ in errors(issues))


def bad_delay_setup(library):
    raw = chain(library).model_dump(mode="python", by_alias=True)
    raw["instances"][2]["settings"] = {"input_delays": {"ch7": "1 ns"}}
    return setup_from_dict(raw)


def test_validation_rejects_tagger_delay_for_missing_channel(tmp_path):
    library = Library(tmp_path / "lib")
    issues = check_setup(bad_delay_setup(library), library)
    assert any(i.code == "invalid_settings" and "ch7" in i.message for i in issues)


def test_rate_engine_rejects_bad_tagger_delay_with_a_clean_error(tmp_path):
    library = Library(tmp_path / "lib")
    with pytest.raises(SetupValidationError, match="ch7"):
        simulate_rates(bad_delay_setup(library), library)


def test_snspd_warns_when_bias_surface_and_setting_do_not_match():
    surface = Surface(
        x=[700.0, 900.0],
        y=[8.0, 12.0],
        z=[[0.1, 0.5], [0.2, 0.9]],
        x_unit="nm",
        x_label="wavelength",
        y_unit="uA",
        y_label="bias",
    )
    with pytest.warns(DetectorConfigWarning, match="no bias_current"):
        make("SNSPD", efficiency=0.3, efficiency_vs_bias=surface).detector_model()
    with pytest.warns(DetectorConfigWarning, match="no efficiency-vs-bias"):
        make("SNSPD", {"bias_current": "10 uA"}, efficiency=0.3).detector_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetectorConfigWarning)
        make("SNSPD", efficiency=0.3).detector_model()
        make(
            "SNSPD", {"bias_current": "10 uA"}, efficiency=0.3, efficiency_vs_bias=surface
        ).detector_model()


def test_rate_engine_surfaces_detector_configuration_warning(tmp_path):
    library = Library(tmp_path / "lib")
    setup = chain(library)
    put(library, "snspd", "SNSPD", efficiency=0.8, efficiency_vs_bias=bias_surface())
    result = simulate_rates(setup, library)
    assert "detector_config" in [i.code for i in result.issues]


def bias_surface():
    return Surface(
        x=[700.0, 900.0],
        y=[8.0, 12.0],
        z=[[0.1, 0.5], [0.2, 0.9]],
        x_unit="nm",
        x_label="wavelength",
        y_unit="uA",
        y_label="bias",
    )


def test_efficiency_is_bounded_to_unit_interval():
    with pytest.raises(ValueError):
        make("SPAD", efficiency=1.2)
    with pytest.raises(ValueError):
        make("SNSPD", efficiency=-0.1)
    curve = Curve(x=[700.0, 900.0], y=[0.5, 1.4], x_unit="nm", x_label="wavelength")
    det = make("SPAD", efficiency=curve)
    with pytest.warns(DetectorConfigWarning, match="clipped"):
        op = operator(det, np.array([700e-9, 900e-9]))
    assert np.allclose(op[:, 0, 0], [0.5, 1.0])
    assert np.allclose(np.linalg.eigvalsh(op), op[:, [0, 1], [0, 1]])


def test_aperture_radius_only_for_free_space_input():
    assert make("SPAD", efficiency=0.5, active_diameter="2 mm").aperture_radius() is None
    free = make("SPAD", {"input": "free_space"}, efficiency=0.5, active_diameter="2 mm")
    assert free.aperture_radius() == pytest.approx(1e-3)
    assert make("SPAD", {"input": "free_space"}, efficiency=0.5).aperture_radius() is None
