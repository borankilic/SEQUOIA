import numpy as np
import pytest

from sequoia.light.gaussian import (
    GaussianMode,
    aperture_transmission,
    coupling_efficiency,
    free_space,
    overlap,
    thin_lens,
)

LAMBDA = 780e-9


def test_rayleigh_range_and_divergence():
    mode = GaussianMode.from_waist(LAMBDA, 0.5e-3)
    assert mode.rayleigh_x == pytest.approx(np.pi * 0.5e-3**2 / LAMBDA)
    assert mode.divergence_x == pytest.approx(LAMBDA / (np.pi * 0.5e-3))
    assert mode.waist_x == pytest.approx(0.5e-3)
    m2 = GaussianMode.from_waist(LAMBDA, 0.5e-3, m2_x=2.0)
    assert m2.divergence_x == pytest.approx(2.0 * LAMBDA / (np.pi * 0.5e-3))
    assert m2.w_x == pytest.approx(0.5e-3)


def test_free_propagation_matches_textbook_formulas():
    w0 = 0.3e-3
    mode = GaussianMode.from_waist(LAMBDA, w0)
    z_r = mode.rayleigh_x
    for z in (0.2, 1.0, 5.0):
        moved = mode.propagate(z)
        assert moved.w_x == pytest.approx(w0 * np.sqrt(1 + (z / z_r) ** 2), rel=1e-12)
        assert moved.radius_x == pytest.approx(z * (1 + (z_r / z) ** 2), rel=1e-12)
        assert moved.waist_x == pytest.approx(w0, rel=1e-12)
        assert moved.distance_past_waist_x == pytest.approx(z)


@pytest.mark.parametrize(("d", "f"), [(0.0, 0.1), (0.3, 0.1), (0.05, 0.2)])
def test_thin_lens_waist_transformation(d, f):
    w0 = 0.4e-3
    mode = GaussianMode.from_waist(LAMBDA, w0, z=d)
    z_r = mode.rayleigh_x
    after = mode.apply(thin_lens(f))
    expected = w0 / np.sqrt((1 - d / f) ** 2 + (z_r / f) ** 2)
    assert after.waist_x == pytest.approx(expected, rel=1e-9)


def test_telescope_magnification():
    mode = GaussianMode.from_waist(LAMBDA, 1.0e-3)
    out = mode.apply(thin_lens(0.05)).propagate(0.05 + 0.1).apply(thin_lens(0.1))
    assert out.w_x / mode.w_x == pytest.approx(2.0, rel=5e-3)
    assert out.divergence_x == pytest.approx(mode.divergence_x / 2.0, rel=5e-3)


def test_astigmatic_lens_acts_on_axes_independently():
    mode = GaussianMode.from_waist(LAMBDA, 2e-3)
    after = mode.apply(thin_lens(0.3), thin_lens(0.6))
    assert -after.distance_past_waist_x == pytest.approx(0.3, rel=5e-3)
    assert -after.distance_past_waist_y == pytest.approx(0.6, rel=5e-3)
    assert after.distance_past_waist_x != pytest.approx(after.distance_past_waist_y, rel=0.1)


def test_abcd_composition_is_associative():
    mode = GaussianMode.from_waist(LAMBDA, 0.5e-3)
    chain = free_space(0.1) @ thin_lens(0.05)
    assert mode.apply(thin_lens(0.05)).propagate(0.1).q_x == pytest.approx(mode.apply(chain).q_x)


def test_overlap_of_matched_and_mismatched_modes():
    a = GaussianMode.from_waist(LAMBDA, 2.5e-6)
    assert coupling_efficiency(a, a) == pytest.approx(1.0)
    b = GaussianMode.from_waist(LAMBDA, 5.0e-6)
    assert coupling_efficiency(a, b) == pytest.approx(4.0 / (2.0 + 0.5) ** 2)
    assert abs(overlap(a, b)) ** 2 == pytest.approx(0.64)


def test_overlap_with_curvature_mismatch_matches_closed_form():
    w, r1 = 1.0e-3, 2.0
    inv_q1 = complex(1.0 / r1, -LAMBDA / (np.pi * w**2))
    inv_q2 = complex(0.0, -LAMBDA / (np.pi * w**2))
    m1 = GaussianMode(LAMBDA, 1 / inv_q1, 1 / inv_q1)
    m2 = GaussianMode(LAMBDA, 1 / inv_q2, 1 / inv_q2)
    expected = 4.0 / (4.0 + (np.pi * w * w / LAMBDA) ** 2 * (1 / r1) ** 2)
    assert coupling_efficiency(m1, m2) == pytest.approx(expected, rel=1e-9)


def test_m_squared_and_lateral_offset_reduce_coupling():
    fiber = GaussianMode.from_waist(LAMBDA, 2.5e-6)
    beam = GaussianMode.from_waist(LAMBDA, 2.5e-6, m2_x=2.0, m2_y=1.0)
    assert coupling_efficiency(beam, fiber) == pytest.approx(0.5)
    offset = 1.0e-6
    expected = np.exp(-2 * offset**2 / (2 * 2.5e-6**2))
    assert coupling_efficiency(fiber, fiber, offset) == pytest.approx(expected)


def test_aperture_transmission():
    mode = GaussianMode.from_waist(LAMBDA, 1e-3)
    assert aperture_transmission(np.inf, mode) == 1.0
    assert aperture_transmission(1e-3, mode) == pytest.approx(1 - np.exp(-2))
    assert aperture_transmission(1.5e-3, mode) > 0.98
