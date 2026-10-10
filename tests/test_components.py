import math

import pytest
from optics import (
    LASER,
    basic_library,
    fiber,
    fs,
    inst,
    power,
    probe,
    put,
    run,
)
from scipy import constants

from sequoia.data import Curve
from sequoia.library.store import Library

LAMBDA = 780e-9
P0 = 1e-3


@pytest.fixture
def lib(tmp_path):
    return basic_library(Library(tmp_path / "lib"))


def line(lib, *stages, laser="laser", probe_part="probes/power", probe_position=None):
    instances = [inst("src", laser)]
    links = []
    prev = "src.out"
    for i, (part, settings) in enumerate(stages):
        instances.append(inst(f"s{i}", part, settings))
        links.append(fs(f"l{i}", prev, f"s{i}.in", "0 mm"))
        prev = f"s{i}.out"
    instances.append(inst("end", "pd"))
    last = f"l{len(stages)}"
    links.append(fs(last, prev, "end.in", "0 mm"))
    probe_spec = probe("p", last, probe_part)
    if probe_position is not None:
        probe_spec["settings"] = {"position": probe_position}
    return run(lib, instances, links, [probe_spec])


@pytest.mark.parametrize("theta", [0, 20, 45, 70, 90])
def test_malus_law(lib, theta):
    result = line(lib, ("polarizer", {"angle": f"{theta} deg"}))
    assert power(result) == pytest.approx(P0 * math.cos(math.radians(theta)) ** 2, abs=5e-12)


def test_laser_polarization_angle_and_extinction(lib):
    put(lib, "leaky", "Laser", **{**LASER, "polarization_extinction_ratio": 100})
    instances = [inst("src", "leaky", {"polarization_angle": "30 deg"}), inst("end", "pd")]
    result = run(lib, instances, [fs("l", "src.out", "end.in")], [probe("p", "l", "probes/pol")])
    p = result.probes["p"]
    assert p["dop"] == pytest.approx(99 / 101)
    assert p["azimuth_rad"] == pytest.approx(math.radians(30))


def test_half_wave_plate_rotates_polarization_by_twice_the_angle(lib):
    result = line(lib, ("hwp", {"angle": "10 deg"}), probe_part="probes/pol")
    assert result.probes["p"]["azimuth_rad"] == pytest.approx(math.radians(20))
    assert result.probes["p"]["dop"] == pytest.approx(1.0)


def test_half_wave_plate_between_polarizers(lib):
    aligned = line(lib, ("hwp", {"angle": "22.5 deg"}), ("polarizer", {"angle": "45 deg"}))
    assert power(aligned) == pytest.approx(P0)
    unrotated = line(lib, ("hwp", {"angle": "0 deg"}), ("polarizer", {"angle": "45 deg"}))
    assert power(unrotated) == pytest.approx(P0 / 2)


def test_quarter_wave_plate_at_45_makes_circular_light(lib):
    result = line(lib, ("qwp", {"angle": "45 deg"}), probe_part="probes/pol")
    p = result.probes["p"]
    assert abs(p["S3"]) == pytest.approx(1.0)
    assert p["dop"] == pytest.approx(1.0)
    assert abs(p["ellipticity_rad"]) == pytest.approx(math.pi / 4)


def test_wave_plate_retardance_scales_with_wavelength(lib):
    put(lib, "qwp_800", "WavePlate", retardance=0.25, design_wavelength="800 nm")
    result = line(lib, ("qwp_800", {"angle": "45 deg"}), probe_part="probes/pol")
    delta = 2 * math.pi * 0.25 * 800 / 780
    assert abs(result.probes["p"]["S3"]) == pytest.approx(abs(math.sin(delta)), abs=1e-7)


def test_pbs_splits_by_polarization(lib):
    def split(angle):
        instances = [
            inst("src", "laser", {"polarization_angle": angle}),
            inst("pbs", "pbs"),
            inst("t", "pd"),
            inst("r", "pd"),
        ]
        links = [
            fs("in", "src.out", "pbs.in"),
            fs("lt", "pbs.out_t", "t.in"),
            fs("lr", "pbs.out_r", "r.in"),
        ]
        probes = [probe("t", "lt"), probe("r", "lr")]
        result = run(lib, instances, links, probes)
        return power(result, "t"), power(result, "r")

    assert split("0 deg") == pytest.approx((P0, 0.0), abs=1e-12)
    assert split("90 deg") == pytest.approx((0.0, P0), abs=1e-12)
    assert split("45 deg") == pytest.approx((P0 / 2, P0 / 2))


def test_pbs_leakage_from_part_data(lib):
    put(lib, "leaky_pbs", "PBS", transmittance_p=0.95, transmittance_s=0.002, reflectance_s=0.99)
    instances = [inst("src", "laser", {"polarization_angle": "90 deg"}), inst("p", "leaky_pbs")]
    instances += [inst("t", "pd"), inst("r", "pd")]
    links = [
        fs("in", "src.out", "p.in"),
        fs("lt", "p.out_t", "t.in"),
        fs("lr", "p.out_r", "r.in"),
    ]
    result = run(lib, instances, links, [probe("t", "lt"), probe("r", "lr")])
    assert power(result, "t") == pytest.approx(P0 * 0.002)
    assert power(result, "r") == pytest.approx(P0 * 0.99)


def test_nd_filter(lib):
    put(lib, "nd2", "NDFilter", optical_density=2.0)
    assert power(line(lib, ("nd2", {}))) == pytest.approx(P0 * 0.01)


def test_spectral_filter_and_angle_tuning(lib):
    ramp = Curve(x=[700.0, 900.0], y=[0.0, 1.0], x_unit="nm", x_label="wavelength")
    put(lib, "ramp", "SpectralFilter", transmission=ramp)
    assert power(line(lib, ("ramp", {}))) == pytest.approx(P0 * 0.4)
    shift = math.sqrt(1 - (math.sin(math.radians(30)) / 2.0) ** 2)
    expected = (780 / shift - 700) / 200
    tilted = line(lib, ("ramp", {"angle_of_incidence": "30 deg"}))
    assert power(tilted) == pytest.approx(P0 * expected)


def test_dichroic_conserves_power(lib):
    put(lib, "dich", "DichroicMirror", transmittance=0.3)
    instances = [inst("src", "laser"), inst("d", "dich"), inst("t", "pd"), inst("r", "pd")]
    links = [
        fs("in", "src.out", "d.in"),
        fs("lt", "d.out_t", "t.in"),
        fs("lr", "d.out_r", "r.in"),
    ]
    result = run(lib, instances, links, [probe("t", "lt"), probe("r", "lr")])
    assert power(result, "t") == pytest.approx(0.3 * P0)
    assert power(result, "r") == pytest.approx(0.7 * P0)


def test_telescope_magnifies_the_beam(lib):
    put(lib, "f50", "Lens", focal_length="50 mm")
    put(lib, "f100", "Lens", focal_length="100 mm")
    instances = [inst("src", "laser"), inst("a", "f50"), inst("b", "f100"), inst("end", "pd")]
    links = [
        fs("l0", "src.out", "a.in", "0 mm"),
        fs("l1", "a.out", "b.in", "150 mm"),
        fs("l2", "b.out", "end.in", "0 mm"),
    ]
    probes = [
        {**probe("b", "l2", "probes/beam"), "settings": {"position": "0 mm"}},
        {**probe("a", "l0", "probes/beam"), "settings": {"position": "0 mm"}},
    ]
    result = run(lib, instances, links, probes)
    ratio = result.probes["b"]["w_x_m"] / result.probes["a"]["w_x_m"]
    assert ratio == pytest.approx(2.0, rel=5e-3)


def test_curved_mirror_astigmatism_at_45_degrees(lib):
    put(lib, "curved", "Mirror", radius_of_curvature="1 m")
    laser = put(lib, "wide", "Laser", **{**LASER, "waist_radius": "3 mm"})
    result = line(
        lib,
        ("curved", {"angle_of_incidence": "45 deg"}),
        laser=laser,
        probe_part="probes/beam",
        probe_position="0 mm",
    )
    p = result.probes["p"]
    assert -p["distance_past_waist_x_m"] == pytest.approx(0.5 * math.cos(math.pi / 4), rel=5e-3)
    assert -p["distance_past_waist_y_m"] == pytest.approx(0.5 / math.cos(math.pi / 4), rel=5e-3)


def test_mirror_reflectance_and_clipping_warning(lib):
    put(lib, "lossy", "Mirror", reflectance=0.9, clear_aperture="1 mm")
    result = line(lib, ("lossy", {}))
    clip = 1 - math.exp(-2 * 0.5**2)
    assert power(result) == pytest.approx(P0 * 0.9 * clip, rel=1e-6)
    assert [i.code for i in result.issues] == ["aperture_clipping"]


def test_iris_clips_gaussian_beam(lib):
    result = line(lib, ("iris", {"diameter": "4 mm"}))
    assert power(result) == pytest.approx(P0 * (1 - math.exp(-2 * 4)), rel=1e-6)
    assert not result.issues


def launch_setup(lib, coupler_settings=None, fiber_part="fiber", length="1 m", **fiber_extra):
    instances = [
        inst("src", "laser"),
        inst("launch", "coupler", coupler_settings),
        inst("recv", "coupler", {"mode": "to_free"}),
        inst("end", "pd"),
    ]
    links = [
        fs("s0", "src.out", "launch.in", "0 mm"),
        fiber("f", "launch.out", "recv.in", fiber_part, length, **fiber_extra),
        fs("s1", "recv.out", "end.in", "0 mm"),
    ]
    return run(lib, instances, links, [probe("p", "f")])


@pytest.fixture
def fiber_lib(lib):
    put(lib, "coupler", "FiberCoupler", focal_length="11 mm")
    put(lib, "fiber", "OpticalFiber", mode_field_diameter="5 um", cutoff_wavelength="730 nm")
    return lib


def test_fiber_coupling_matches_the_gaussian_overlap_formula(fiber_lib):
    result = launch_setup(fiber_lib)
    w_in, f, w_fiber = 1e-3, 11e-3, 2.5e-6
    z_r = math.pi * w_in**2 / LAMBDA
    w_focus = w_in / math.sqrt(1 + (z_r / f) ** 2)
    expected = 4 / (w_focus / w_fiber + w_fiber / w_focus) ** 2
    assert result.links["f"].info["coupling_efficiency"] == pytest.approx(expected, rel=2e-3)


def test_collimator_recovers_a_collimated_beam(fiber_lib):
    result = launch_setup(fiber_lib)
    out = result.links["s1"].end.mode
    assert out.w_x == pytest.approx(LAMBDA * 11e-3 / (math.pi * 2.5e-6), rel=1e-4)
    assert abs(out.radius_x) > 500.0
    assert out.divergence_x < 1e-3


def test_fiber_attenuation_and_connector_loss(fiber_lib):
    put(
        fiber_lib,
        "lossy_fiber",
        "OpticalFiber",
        mode_field_diameter="5 um",
        attenuation_db_per_km=3.0103,
    )
    result = launch_setup(
        fiber_lib, fiber_part="lossy_fiber", length="1 km", connector_loss_db=3.0103
    )
    link = result.links["f"]
    assert link.info["transmission"] == pytest.approx(0.25, rel=1e-4)
    expected = link.start.power * link.info["coupling_efficiency"] * 0.25
    assert link.end.power == pytest.approx(expected, rel=1e-4)


def test_fiber_below_cutoff_is_flagged(fiber_lib):
    put(
        fiber_lib,
        "short_cutoff",
        "OpticalFiber",
        mode_field_diameter="5 um",
        cutoff_wavelength="850 nm",
    )
    result = launch_setup(fiber_lib, fiber_part="short_cutoff")
    assert "fiber_multimode" in [i.code for i in result.issues]
    assert "fiber_multimode" not in [i.code for i in launch_setup(fiber_lib).issues]


def test_multimode_warning_is_suppressed_at_negligible_power():
    from types import SimpleNamespace

    from sequoia.components.base import Context, get_component
    from sequoia.data import Scalar

    link = SimpleNamespace(
        length=Scalar(value=1.0, unit="m"),
        connector_loss_db=0.0,
        polarization=SimpleNamespace(kind="identity"),
    )

    def codes(power):
        laser_cls = get_component("Laser")
        laser = laser_cls(laser_cls.Params(**{**LASER, "power": power}))
        ctx = Context(instance_id="src")
        beam = laser.emit(ctx).beams[0][1]
        fiber_cls = get_component("OpticalFiber")
        part = fiber_cls(fiber_cls.Params(mode_field_diameter="5 um", cutoff_wavelength="850 nm"))
        part.transmit(beam, link, ctx, "f")
        return [i.code for i in ctx.issues]

    quiet = codes("1 pW")
    assert quiet == []
    loud = codes("1 mW")
    assert loud == ["fiber_multimode"]


def test_context_power_threshold_is_configurable():
    from sequoia.components.base import Context

    assert Context().min_power_for_warnings == 1e-9


def test_poor_coupling_and_defocus_are_flagged(fiber_lib):
    good = launch_setup(fiber_lib).links["f"].info["coupling_efficiency"]
    result = launch_setup(fiber_lib, {"defocus": "1 mm"})
    assert result.links["f"].info["coupling_efficiency"] < 0.1 * good
    assert "low_coupling" in [i.code for i in result.issues]


def test_lateral_offset_reduces_coupling(fiber_lib):
    reference = launch_setup(fiber_lib)
    shifted = launch_setup(fiber_lib, {"lateral_offset": "2 um"})
    w_face = reference.links["f"].start.mode.w_x
    factor = math.exp(-2 * (2e-6) ** 2 / (w_face**2 + 2.5e-6**2))
    ratio = (
        shifted.links["f"].info["coupling_efficiency"]
        / reference.links["f"].info["coupling_efficiency"]
    )
    assert ratio == pytest.approx(factor, rel=1e-9)


def test_fiber_polarization_transforms(fiber_lib):
    def final_power(polarization):
        put(fiber_lib, "pol_out", "Polarizer", extinction_ratio=1e9)
        instances = [
            inst("src", "laser"),
            inst("launch", "coupler"),
            inst("recv", "coupler", {"mode": "to_free"}),
            inst("pol", "pol_out", {"angle": "90 deg"}),
            inst("end", "pd"),
        ]
        links = [
            fs("s0", "src.out", "launch.in", "0 mm"),
            fiber("f", "launch.out", "recv.in", "fiber", "1 m", polarization=polarization),
            fs("s1", "recv.out", "pol.in", "0 mm"),
            fs("s2", "pol.out", "end.in", "0 mm"),
        ]
        result = run(fiber_lib, instances, links, [probe("through", "s2"), probe("all", "s1")])
        return power(result, "through"), power(result, "all")

    through, total = final_power({"kind": "identity"})
    assert through == pytest.approx(0.0, abs=1e-11)
    flipped, total_flipped = final_power({"kind": "su2", "rotation_vector": [math.pi, 0, 0]})
    assert flipped == pytest.approx(total_flipped, rel=1e-9)
    random_through, random_total = final_power({"kind": "random", "seed": 3})
    assert random_total == pytest.approx(total, rel=1e-9)
    again, _ = final_power({"kind": "random", "seed": 3})
    assert again == pytest.approx(random_through)
    assert 0.0 < random_through < random_total


def test_paddle_retardance_follows_the_bend_birefringence_formula(fiber_lib):
    put(fiber_lib, "paddles", "FiberPaddleController", loops=[3])
    instances = [
        inst("src", "laser", {"polarization_angle": "45 deg"}),
        inst("launch", "coupler"),
        inst("fpc", "paddles"),
        inst("recv", "coupler", {"mode": "to_free"}),
        inst("end", "pd"),
    ]
    links = [
        fs("s0", "src.out", "launch.in", "0 mm"),
        fiber("f1", "launch.out", "fpc.in", "fiber", "0.5 m"),
        fiber("f2", "fpc.out", "recv.in", "fiber", "0.5 m"),
        fs("s1", "recv.out", "end.in", "0 mm"),
    ]
    result = run(fiber_lib, instances, links, [probe("p", "s1", "probes/pol")])
    delta = 2 * math.pi**2 * 0.133 * (125e-6) ** 2 * 3 / (LAMBDA * 56e-3)
    p = result.probes["p"]
    assert p["S2"] == pytest.approx(math.cos(delta), abs=1e-6)
    assert abs(p["S3"]) == pytest.approx(abs(math.sin(delta)), abs=1e-6)
    assert p["dop"] == pytest.approx(1.0, abs=1e-6)


def test_fiber_beam_splitter_divides_power(fiber_lib):
    put(fiber_lib, "fbs", "FiberBeamSplitter", split_ratio=0.3, transmission=0.9)
    instances = [
        inst("src", "laser"),
        inst("launch", "coupler"),
        inst("split", "fbs"),
        inst("ra", "coupler", {"mode": "to_free"}),
        inst("rb", "coupler", {"mode": "to_free"}),
        inst("pa", "pd"),
        inst("pb", "pd"),
    ]
    links = [
        fs("s0", "src.out", "launch.in", "0 mm"),
        fiber("f0", "launch.out", "split.in_a", "fiber"),
        fiber("fa", "split.out_a", "ra.in", "fiber"),
        fiber("fb", "split.out_b", "rb.in", "fiber"),
        fs("sa", "ra.out", "pa.in"),
        fs("sb", "rb.out", "pb.in"),
    ]
    result = run(fiber_lib, instances, links, [probe("pa", "fa"), probe("pb", "fb")])
    assert power(result, "pb") / power(result, "pa") == pytest.approx(0.3 / 0.7, rel=1e-6)
    f0_end = result.links["f0"].end.power
    assert power(result, "pa") + power(result, "pb") == pytest.approx(0.9 * f0_end, rel=1e-9)


def test_photodiode_photocurrent_and_overfill_warning(lib):
    result = line(lib, ("mirror", {}))
    assert result.readouts["end"]["photocurrent_A"] == pytest.approx(P0 * 0.5 + 2e-9)
    put(lib, "tiny_pd", "Photodiode", responsivity="0.5 A/W", active_diameter="1 mm")
    instances = [inst("src", "laser"), inst("end", "tiny_pd")]
    small = run(lib, instances, [fs("l", "src.out", "end.in")])
    assert small.readouts["end"]["collected_fraction"] == pytest.approx(
        1 - math.exp(-2 * 0.25), rel=1e-3
    )
    assert "detector_overfilled" in [i.code for i in small.issues]


def test_laser_power_from_current_curve_and_setting_override(lib):
    ramp = Curve(x=[0.0, 0.1], y=[0.0, 0.1], x_unit="A", y_unit="W", x_label="current")
    put(lib, "diode", "Laser", **{**LASER, "power_vs_current": ramp})
    instances = [inst("src", "diode", {"current": "0.06 A"}), inst("end", "pd")]
    links = [fs("l", "src.out", "end.in")]
    assert power(run(lib, instances, links, [probe("p", "l")])) == pytest.approx(0.06)
    instances = [inst("src", "diode", {"power": "3 mW"}), inst("end", "pd")]
    assert power(run(lib, instances, links, [probe("p", "l")])) == pytest.approx(3e-3)


def test_pulsed_laser_photons_per_pulse_and_bandwidth(lib):
    put(
        lib,
        "fs_laser",
        "Laser",
        **{**LASER, "repetition_rate": "80 MHz", "pulse_width": "100 fs"},
    )
    instances = [inst("src", "fs_laser"), inst("end", "pd")]
    links = [fs("l", "src.out", "end.in")]
    put(lib, "probes/spectrum", "Spectrometer")
    probes = [probe("p", "l"), probe("s", "l", "probes/spectrum")]
    result = run(lib, instances, links, probes)
    photon_energy = constants.h * constants.c / LAMBDA
    expected = P0 / (80e6 * photon_energy)
    assert result.probes["p"]["mean_photons_per_pulse"] == pytest.approx(expected, rel=2e-3)
    fwhm = 0.441 / 100e-15 * LAMBDA**2 / constants.c
    assert result.probes["s"]["rms_width_m"] == pytest.approx(fwhm / 2.355, rel=0.15)
    assert result.probes["s"]["centroid_wavelength_m"] == pytest.approx(LAMBDA, rel=1e-3)


def test_curve_extrapolation_is_reported_as_a_diagnostic(lib):
    narrow = Curve(x=[700.0, 750.0], y=[0.5, 0.6], x_unit="nm", x_label="wavelength")
    put(lib, "narrow", "SpectralFilter", transmission=narrow)
    result = line(lib, ("narrow", {}))
    assert "curve_extrapolation" in [i.code for i in result.issues]
    assert power(result) == pytest.approx(P0 * 0.6)


def mzi(lib, extra_arm: str, laser: str = "laser"):
    instances = [
        inst("src", laser),
        inst("bs1", "bs"),
        inst("m1", "mirror"),
        inst("m2", "mirror"),
        inst("bs2", "bs"),
        inst("d1", "pd"),
        inst("d2", "pd"),
    ]
    links = [
        fs("in", "src.out", "bs1.in_a"),
        fs("a1", "bs1.out_a", "m1.in"),
        fs("a2", "m1.out", "bs2.in_a"),
        fs("b1", "bs1.out_b", "m2.in", extra_arm),
        fs("b2", "m2.out", "bs2.in_b"),
        fs("o1", "bs2.out_a", "d1.in"),
        fs("o2", "bs2.out_b", "d2.in"),
    ]
    return run(lib, instances, links, [probe("p1", "o1"), probe("p2", "o2")])


def test_balanced_mzi_sends_everything_to_one_port(lib):
    result = mzi(lib, "100 mm")
    assert power(result, "p2") == pytest.approx(P0)
    assert power(result, "p1") == pytest.approx(0.0, abs=1e-12)


def test_mzi_fringe_follows_the_path_difference(lib):
    for quarter in range(5):
        delta = quarter * LAMBDA / 4
        result = mzi(lib, f"{0.1 + delta!r} m")
        phase = 2 * math.pi * delta / LAMBDA
        assert power(result, "p2") == pytest.approx(P0 * math.cos(phase / 2) ** 2, abs=1e-9)
        assert power(result, "p1") + power(result, "p2") == pytest.approx(P0)


def test_mzi_visibility_equals_the_laser_coherence_function(lib):
    put(lib, "narrow_laser", "Laser", **{**LASER, "linewidth": "150 MHz"})
    arm = 0.3
    powers = []
    for k in range(4):
        result = mzi(lib, f"{0.1 + arm + k * LAMBDA / 4!r} m", laser="narrow_laser")
        powers.append(power(result, "p2"))
    visibility = math.hypot(powers[0] - powers[2], powers[1] - powers[3]) / (powers[0] + powers[2])
    tau = arm / constants.c
    assert visibility == pytest.approx(math.exp(-math.pi * 150e6 * tau), rel=1e-2)


def test_different_lasers_add_incoherently(lib):
    instances = [
        inst("a", "laser"),
        inst("b", "laser", {"power": "3 mW"}),
        inst("bs", "bs"),
        inst("d", "pd"),
    ]
    links = [
        fs("la", "a.out", "bs.in_a"),
        fs("lb", "b.out", "bs.in_b"),
        fs("lo", "bs.out_a", "d.in"),
    ]
    result = run(lib, instances, links, [probe("p", "lo")])
    assert power(result) == pytest.approx(0.5 * (P0 + 3e-3))
    assert result.links["lo"].end.sources == ("a", "b")
