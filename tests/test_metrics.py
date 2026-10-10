import numpy as np
import pytest

from sequoia.analysis import metrics as m

RHO_PHI = m.werner_state(1.0)


def test_heralding_car_visibility_g2_hand_numbers():
    assert m.heralding_efficiency(300.0, 1000.0) == pytest.approx(0.3)
    assert m.heralding_efficiency(310.0, 1000.0, accidentals=10.0) == pytest.approx(0.3)
    acc = m.accidentals_cw(1e5, 2e5, 1e-9)
    assert acc == pytest.approx(20.0)
    assert m.car(500.0, acc) == pytest.approx(25.0)
    assert m.car_from_total(520.0, acc) == pytest.approx(25.0)
    assert m.visibility(90.0, 10.0) == pytest.approx(0.8)
    assert m.g2_heralded(2.0, 1000.0, 400.0, 500.0) == pytest.approx(0.01)
    assert m.g2_from_schmidt(4.0) == pytest.approx(1.25)


def test_visibility_from_phases():
    phi0 = 0.7
    v = 0.83
    c = [100 * (1 + v * np.cos(p + phi0)) for p in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
    assert m.visibility_from_phases(*c) == pytest.approx(v)


def test_state_measures():
    ket = m.bell_state("psi-")
    prod = np.kron(m.KETS["H"], m.KETS["D"])
    assert m.concurrence(ket) == pytest.approx(1.0)
    assert m.concurrence(prod) == pytest.approx(0.0, abs=1e-12)
    assert m.purity(ket) == pytest.approx(1.0)
    assert m.purity(np.eye(4) / 4) == pytest.approx(0.25)
    assert m.fidelity(ket, ket) == pytest.approx(1.0)
    assert m.fidelity(m.bell_state("phi+"), m.bell_state("psi+")) == pytest.approx(0.0, abs=1e-12)
    v = 0.6
    w = m.werner_state(v)
    expected_purity = ((3 * v + 1) / 4) ** 2 + 3 * ((1 - v) / 4) ** 2
    assert m.purity(w) == pytest.approx(expected_purity)
    assert m.fidelity(w, m.bell_state()) == pytest.approx((3 * v + 1) / 4)
    assert m.concurrence(w) == pytest.approx(max(0.0, (3 * v - 1) / 2))
    assert m.concurrence(m.werner_state(0.2)) == 0.0


def test_fidelity_mixed_mixed():
    a = np.diag([0.5, 0.5, 0, 0]).astype(complex)
    b = np.diag([0.5, 0, 0.5, 0]).astype(complex)
    assert m.fidelity(a, b) == pytest.approx(0.25)


@pytest.mark.parametrize("v", [1.0, 0.9, 0.5])
def test_chsh_werner(v):
    s = m.chsh_from_state(m.werner_state(v))
    assert s == pytest.approx(2 * np.sqrt(2) * v)


def test_chsh_psi_minus_and_classical_bound():
    th = np.deg2rad(m.CHSH_ANGLES_DEG)
    e = m.correlation_from_angles
    rho = m.werner_state(1.0, "psi-")
    assert e(rho, 0.3, 0.3) == pytest.approx(-1.0)
    assert e(rho, 0.0, np.pi / 2) == pytest.approx(1.0)
    s = m.chsh_from_state(rho, (th[0], th[1], th[2] + np.pi / 4, th[3] + np.pi / 4))
    assert abs(s) <= 2 * np.sqrt(2) + 1e-12
    prod = np.outer(np.kron(m.KETS["H"], m.KETS["H"]), np.kron(m.KETS["H"], m.KETS["H"]))
    sp = max(
        abs(m.chsh_from_state(prod, tuple(np.random.default_rng(i).uniform(0, np.pi, 4))))
        for i in range(200)
    )
    assert sp <= 2.0 + 1e-12


def test_correlation_phi_plus_analytic():
    for ta, tb in [(0.1, 0.4), (1.0, -0.3), (0.0, 0.0)]:
        assert m.correlation_from_angles(RHO_PHI, ta, tb) == pytest.approx(np.cos(2 * (ta - tb)))


def test_chsh_from_rates_matches_state():
    rho = m.werner_state(0.8)
    a = np.deg2rad([0.0, 45.0])
    b = np.deg2rad([22.5, 67.5])
    rates = np.zeros((2, 2, 2, 2))
    perp = np.pi / 2
    for i in range(2):
        for j in range(2):
            for oa in range(2):
                for ob in range(2):
                    ka = m.linear_ket(a[i] + oa * perp)
                    kb = m.linear_ket(b[j] + ob * perp)
                    p = np.kron(ka, kb)
                    rates[i, j, oa, ob] = 1e4 * np.real(p.conj() @ rho @ p)
    assert m.chsh_from_rates(rates) == pytest.approx(2 * np.sqrt(2) * 0.8)


def test_schmidt_matrix_cases():
    prod = np.outer([1, 2, 3.0], [1, -1, 0.5])
    assert m.schmidt_number_matrix(prod) == pytest.approx(1.0)
    for d in (2, 3, 5):
        assert m.schmidt_number_matrix(np.eye(d) / np.sqrt(d)) == pytest.approx(d)
    assert m.schmidt_number_matrix(np.diag([0.8, 0.6])) == pytest.approx(1 / (0.64**2 + 0.36**2))


def test_schmidt_amplitude_polarization_and_frequency():
    amp = np.zeros((2, 2, 1, 1), dtype=complex)
    amp[0, 0, 0, 0] = amp[1, 1, 0, 0] = 1 / np.sqrt(2)
    assert m.schmidt_number(amp, [0], [0]) == pytest.approx(2.0)
    amp = np.zeros((2, 2, 1, 1), dtype=complex)
    amp[0, 1, 0, 0] = 1.0
    assert m.schmidt_number(amp, [0], [0]) == pytest.approx(1.0)

    d = 4
    n_signal = np.arange(d)
    n_sum = np.array([2 * d])
    amp = np.zeros((2, 2, d, 1), dtype=complex)
    amp[0, 0, :, 0] = 1 / np.sqrt(d)
    assert m.schmidt_number(amp, n_signal, n_sum) == pytest.approx(d)
    with pytest.raises(ValueError, match="n_signal and n_sum"):
        m.schmidt_number(amp)


def test_schmidt_gaussian_jsa_analytic():
    n = np.arange(-60, 61)
    a, b = 3.0, 8.0
    ns, ni = np.meshgrid(n, n, indexing="ij")
    jsa = np.exp(-((ns + ni) ** 2) / (2 * a**2) - ((ns - ni) ** 2) / (2 * b**2))
    jsa /= np.linalg.norm(jsa)
    n_sum = np.arange(-120, 121)
    amp = np.zeros((2, 2, len(n), len(n_sum)), dtype=complex)
    for j, nj in enumerate(n):
        for mm, nm in enumerate(n_sum):
            ni_ = nm - nj
            if -60 <= ni_ <= 60:
                amp[0, 0, j, mm] = jsa[j, ni_ + 60]
    k = m.schmidt_number(amp, n, n_sum)
    assert k == pytest.approx((a**2 + b**2) / (2 * a * b), rel=1e-3)


def test_tomography_settings_projectors():
    s = m.tomography_settings()
    assert len(s) == 16
    for first in ("hwp", "qwp"):
        for st in m.tomography_settings(first):
            exp = np.kron(m.KETS[st.label[0]], m.KETS[st.label[1]])
            assert abs(np.vdot(exp, st.ket)) == pytest.approx(1.0)
            assert np.trace(st.projector) == pytest.approx(1.0)
    ang = {st.label: (st.hwp_a, st.qwp_a) for st in s}
    assert ang["HH"] == (0.0, 0.0)
    assert np.rad2deg(ang["VV"][0]) == pytest.approx(45.0)
    assert np.rad2deg(ang["DD"][0]) == pytest.approx(22.5)
    total = sum(st.projector for st in s)
    assert np.linalg.matrix_rank(total) == 4
    basis = np.array([st.projector.ravel() for st in s])
    assert np.linalg.matrix_rank(basis) == 16


def test_tomography_counts_scaling_and_noise():
    s = m.tomography_settings()
    rng = np.random.default_rng(1)
    rho = m.werner_state(1.0)
    exp = 1e6 * np.array([np.real(np.trace(st.projector @ rho)) for st in s])
    c = m.tomography_counts(rho, s, 1e5, 10.0, rng)
    assert np.all(np.abs(c - exp) < 6 * np.sqrt(exp) + 1)
    assert exp[0] == pytest.approx(5e5)
    assert c[1] == 0


@pytest.mark.parametrize("first", ["hwp", "qwp"])
@pytest.mark.parametrize("name", ["phi+", "psi-"])
def test_mle_noiseless_recovers_state(first, name):
    s = m.tomography_settings(first)
    rho = m.werner_state(0.85, name)
    counts = m.tomography_probabilities(rho, s) * 1e9
    est = m.mle_density_matrix(counts, s)
    assert np.max(np.abs(est - rho)) < 1e-6


def test_mle_noiseless_pure_and_arbitrary_state():
    s = m.tomography_settings()
    rng = np.random.default_rng(5)
    g = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = g @ g.conj().T
    rho /= np.trace(rho)
    est = m.mle_density_matrix(1e6 * m.tomography_probabilities(rho, s), s)
    assert np.max(np.abs(est - rho)) < 1e-6
    pure = m.bell_state("psi+")
    est = m.mle_density_matrix(1e6 * m.tomography_probabilities(pure, s), s)
    assert m.fidelity(est, pure) > 1 - 1e-6


def test_mle_poisson_werner():
    s = m.tomography_settings()
    rho = m.werner_state(0.9)
    rng = np.random.default_rng(42)
    counts = m.tomography_counts(rho, s, 2e4, 1.0, rng)
    assert counts.sum() > 4e4
    est = m.mle_density_matrix(counts, s)
    assert m.fidelity(est, rho) > 0.99
    assert abs(m.concurrence(est) - m.concurrence(rho)) < 0.05


def test_mle_always_valid_with_noisy_and_sparse_counts():
    s = m.tomography_settings()
    rng = np.random.default_rng(3)
    for rate in (5.0, 50.0, 500.0):
        counts = m.tomography_counts(m.werner_state(1.0), s, rate, 1.0, rng)
        est = m.mle_density_matrix(counts, s)
        assert np.trace(est).real == pytest.approx(1.0)
        assert np.allclose(est, est.conj().T)
        assert np.linalg.eigvalsh(est).min() > -1e-12
    est = m.mle_density_matrix(np.zeros(16), s)
    assert np.trace(est).real == pytest.approx(1.0)


def test_mle_incomplete_settings_rejected():
    s = m.tomography_settings()[:3]
    with pytest.raises(ValueError):
        m.mle_density_matrix(np.ones(3), s)
