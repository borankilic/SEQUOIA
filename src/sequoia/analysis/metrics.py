"""Measurement metrics and two-qubit polarization tomography.

Conventions
-----------
Two-qubit basis order is |HH>, |HV>, |VH>, |VV>; the first qubit is the signal (analyser a), the
second the idler (analyser b). Polarization kets are in the (H, V) basis with
D = (H+V)/sqrt2, A = (H-V)/sqrt2, R = (H-iV)/sqrt2, L = (H+iV)/sqrt2.
Linear analyser angles are measured from H; theta -> ket (cos theta, sin theta).
Wave plates have Jones matrix R(t) diag(1, -1 | i) R(-t) with t the fast-axis angle from H
(retardance added to V, E ~ exp[i(kz - wt)]). An analyser arm is wave plate 1, wave plate 2, then
a polarizer passing H, so it projects onto the ket (W2 W1)^dagger |H>.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

SQRT2 = np.sqrt(2.0)

KETS = {
    "H": np.array([1, 0], dtype=complex),
    "V": np.array([0, 1], dtype=complex),
    "D": np.array([1, 1], dtype=complex) / SQRT2,
    "A": np.array([1, -1], dtype=complex) / SQRT2,
    "R": np.array([1, -1j], dtype=complex) / SQRT2,
    "L": np.array([1, 1j], dtype=complex) / SQRT2,
}

JAMES_ORDER = (
    "HH",
    "HV",
    "VV",
    "VH",
    "RH",
    "RV",
    "DV",
    "DH",
    "DR",
    "DD",
    "RD",
    "HD",
    "VD",
    "VL",
    "HL",
    "RL",
)

BELL = {
    "phi+": np.array([1, 0, 0, 1], dtype=complex) / SQRT2,
    "phi-": np.array([1, 0, 0, -1], dtype=complex) / SQRT2,
    "psi+": np.array([0, 1, 1, 0], dtype=complex) / SQRT2,
    "psi-": np.array([0, 1, -1, 0], dtype=complex) / SQRT2,
}


def ratio(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    return float("nan") if numerator <= 0 else float("inf")


def car(coincidences, accidentals):
    return coincidences / accidentals


def car_from_total(total_coincidences, accidentals):
    return total_coincidences / accidentals - 1.0


def accidentals_cw(singles_a, singles_b, window):
    return singles_a * singles_b * window


def heralding_efficiency(coincidences, singles_herald, accidentals=0.0):
    return (coincidences - accidentals) / singles_herald


def visibility(vmax, vmin):
    return (vmax - vmin) / (vmax + vmin)


def visibility_from_phases(c0, c1, c2, c3):
    """Visibility from counts at phases 0, pi/2, pi, 3pi/2 of A(1 + V cos(phi + phi0))."""
    return 2.0 * np.hypot(c0 - c2, c1 - c3) / (c0 + c1 + c2 + c3)


def correlation_from_rates(c_pp, c_pm, c_mp, c_mm):
    """E = (C++ + C-- - C+- - C-+) / (sum); '-' is the orthogonal analyser outcome."""
    return (c_pp + c_mm - c_pm - c_mp) / (c_pp + c_mm + c_pm + c_mp)


def chsh_s(e_ab, e_abp, e_apb, e_apbp):
    """S = E(a,b) - E(a,b') + E(a',b) + E(a',b')."""
    return e_ab - e_abp + e_apb + e_apbp


def chsh_from_rates(rates):
    """CHSH S from rates[a_setting, b_setting, outcome_a, outcome_b] (shape (2,2,2,2)).

    Setting 0 is a (b), setting 1 is a' (b'); outcome 0 is the analyser ket, 1 its orthogonal.
    """
    r = np.asarray(rates, dtype=float)
    e = np.empty((2, 2))
    for i, j in product(range(2), range(2)):
        e[i, j] = correlation_from_rates(r[i, j, 0, 0], r[i, j, 0, 1], r[i, j, 1, 0], r[i, j, 1, 1])
    return chsh_s(e[0, 0], e[0, 1], e[1, 0], e[1, 1])


def g2_heralded(triple_rate, herald_singles, double_ha, double_hb):
    """g2_H(0) = R_Hab * S_H / (R_Ha * R_Hb)."""
    return triple_rate * herald_singles / (double_ha * double_hb)


def g2_from_schmidt(k):
    return 1.0 + 1.0 / k


def schmidt_coefficients_matrix(matrix):
    s = np.linalg.svd(np.asarray(matrix, dtype=complex), compute_uv=False)
    lam = s**2
    return lam / lam.sum()


def schmidt_number_matrix(matrix):
    lam = schmidt_coefficients_matrix(matrix)
    return float(1.0 / np.sum(lam**2))


def amplitude_to_matrix(amplitude, n_signal=None, n_sum=None):
    """Reshape a PairSource amplitude (sigma_s, sigma_i, j, m) to a signal x idler matrix.

    Rows index (sigma_s, j) with j the signal frequency index. If n_signal and n_sum are given,
    columns index (sigma_i, n_idler) on the lattice n_idler = n_sum[m] - n_signal[j], so the idler
    frequency is resolved correctly (needed for CW pumps, where m has one entry). Without them,
    columns are (sigma_i, m), which is only valid when m already is the idler frequency index.
    """
    a = np.asarray(amplitude, dtype=complex)
    ps, pi, nj, nm = a.shape
    if n_signal is None or n_sum is None:
        return a.transpose(0, 2, 1, 3).reshape(ps * nj, pi * nm)
    n_signal = np.asarray(n_signal)
    n_sum = np.asarray(n_sum)
    n_idler = n_sum[None, :] - n_signal[:, None]
    values, inverse = np.unique(n_idler, return_inverse=True)
    inverse = inverse.reshape(n_idler.shape)
    out = np.zeros((ps, nj, pi, len(values)), dtype=complex)
    jj = np.arange(nj)[:, None] * np.ones((1, nm), dtype=int)
    for s, i in product(range(ps), range(pi)):
        np.add.at(out[s, :, i, :], (jj, inverse), a[s, i])
    return out.reshape(ps * nj, pi * len(values))


def schmidt_number(amplitude, n_signal=None, n_sum=None):
    """Schmidt number 1/sum(lambda^2) of a pure amplitude (sigma_s, sigma_i, j, m).

    An ensemble amplitude (members, sigma_s, sigma_i, j, m) whose members carry sqrt(weight)
    gives the participation ratio (tr rho)^2 / tr rho^2 of the one-body density matrix of the
    mixture, rho = sum_k M_k M_k^dagger (reduced signal state, partner traced out).

    Meaning of the ensemble number. For a pure amplitude, 1 + 1/K is the unheralded g2 of one
    arm. For a mixture it is the g2 only when the members share a common reduced state (for
    example polarization mixing of one JSA, as in a Werner state). The rate engine assumes, for
    pulsed sources, that every pulse is entirely in one member (a shot-to-shot mixture, so the
    per pulse g2 averages 1 + 1/K_k over the members with their weights), and, for CW sources,
    that independent pairs are drawn from the mixture, i.e. the field is the Gaussian field of
    the one-body density matrix. This number is the one-body (purity) quantity of the second
    model, and is not the g2 of the first when members differ in spectral structure: members
    f1 g1 and f2 g2 with disjoint bands give g2 = 2 per pulse but K = 2 here. A pure multimode
    JSA must never be encoded as an ensemble of its Schmidt modes.
    """
    if (n_signal is None or n_sum is None) and np.shape(amplitude)[-1] == 1:
        raise ValueError(
            "a single sum-frequency entry (CW pump) needs n_signal and n_sum to resolve the idler "
            "frequency; pass the lattice arguments of the PairSource"
        )
    a = np.asarray(amplitude)
    if a.ndim == 5:
        mats = [amplitude_to_matrix(m, n_signal, n_sum) for m in a]
        rho = sum(m @ m.conj().T for m in mats)
        return float(np.real(np.trace(rho)) ** 2 / np.real(np.trace(rho @ rho)))
    return schmidt_number_matrix(amplitude_to_matrix(amplitude, n_signal, n_sum))


def bell_state(name="phi+"):
    return BELL[name].copy()


def werner_state(v, name="phi+"):
    psi = BELL[name]
    return v * np.outer(psi, psi.conj()) + (1.0 - v) * np.eye(4) / 4.0


def _as_density(x):
    x = np.asarray(x, dtype=complex)
    if x.ndim == 1:
        return np.outer(x, x.conj())
    return x


def _psd_sqrt(m):
    w, v = np.linalg.eigh((m + m.conj().T) / 2)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.conj().T


def purity(rho):
    r = _as_density(rho)
    return float(np.real(np.trace(r @ r)))


def fidelity(rho, sigma):
    """Uhlmann fidelity F = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2; kets are accepted."""
    r = _as_density(rho)
    s = _as_density(sigma)
    sr = _psd_sqrt(r)
    w = np.linalg.eigvalsh(sr @ s @ sr)
    return float(np.sum(np.sqrt(np.clip(w, 0.0, None))) ** 2)


def concurrence(rho):
    """Wootters concurrence of a two-qubit state (ket or density matrix)."""
    r = _as_density(rho)
    sy = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sy, sy)
    rt = r @ (yy @ r.conj() @ yy)
    lam = np.sqrt(np.clip(np.sort(np.linalg.eigvals(rt).real)[::-1], 0.0, None))
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def _rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]], dtype=complex)


def hwp_jones(theta):
    return _rot(theta) @ np.diag([1.0, -1.0]) @ _rot(-theta)


def qwp_jones(theta):
    return _rot(theta) @ np.diag([1.0, 1.0j]) @ _rot(-theta)


def analyzer_ket(hwp, qwp, first="hwp"):
    """Ket projected onto by wave plates (hwp, qwp angles in rad) followed by an H polarizer."""
    w = qwp_jones(qwp) @ hwp_jones(hwp) if first == "hwp" else hwp_jones(hwp) @ qwp_jones(qwp)
    return w.conj().T @ KETS["H"]


def _same_ray(a, b):
    return abs(abs(np.vdot(a, b)) - 1.0) < 1e-9


def single_qubit_angles(label, first="hwp"):
    """Smallest (hwp, qwp) angles in rad, multiples of 22.5 deg, that project onto label."""
    target = KETS[label]
    grid = np.deg2rad(np.arange(-90.0, 90.0, 22.5))
    best = None
    for h, q in product(grid, grid):
        if _same_ray(analyzer_ket(h, q, first), target):
            cost = abs(h) + abs(q) + 1e-6 * ((h < 0) + (q < 0))
            if best is None or cost < best[0] - 1e-12:
                best = (cost, h, q)
    return best[1], best[2]


@dataclass(frozen=True)
class TomographySetting:
    label: str
    hwp_a: float
    qwp_a: float
    hwp_b: float
    qwp_b: float
    ket_a: np.ndarray
    ket_b: np.ndarray

    @property
    def ket(self):
        return np.kron(self.ket_a, self.ket_b)

    @property
    def projector(self):
        k = self.ket
        return np.outer(k, k.conj())


def tomography_settings(first="hwp", labels=JAMES_ORDER):
    """The 16 James et al. two-qubit settings with the wave plate angles that realise them."""
    out = []
    for lab in labels:
        ha, qa = single_qubit_angles(lab[0], first)
        hb, qb = single_qubit_angles(lab[1], first)
        out.append(
            TomographySetting(
                lab,
                ha,
                qa,
                hb,
                qb,
                analyzer_ket(ha, qa, first),
                analyzer_ket(hb, qb, first),
            )
        )
    return out


def tomography_probabilities(rho, settings):
    r = _as_density(rho)
    return np.array([np.real(np.trace(s.projector @ r)) for s in settings])


def tomography_counts(rho, settings, total_rate, integration_time, rng):
    """Poisson counts; the mean of setting k is total_rate * integration_time * tr(P_k rho).

    total_rate is the pair rate arriving at the two analysers before projection.
    """
    mean = total_rate * integration_time * np.clip(tomography_probabilities(rho, settings), 0, None)
    return rng.poisson(mean).astype(float)


def mle_density_matrix(counts, settings, max_iter=20000, tol=1e-10):
    """Maximum-likelihood density matrix from tomography counts (iterative R rho R).

    The overall rate is a free nuisance parameter. Writing Z = sum_k P_k and E_k = Z^-1/2 P_k Z^-1/2
    makes {E_k} a POVM, so the multinomial likelihood in sigma = Z^1/2 rho Z^1/2 / tr(...) is
    maximised with the Hradil R rho R iteration and mapped back. The result is positive semidefinite
    with unit trace by construction. Iteration stops when the largest element change of sigma is
    below tol.
    """
    n = np.asarray(counts, dtype=float)
    projectors = np.array([s.projector for s in settings])
    z = projectors.sum(axis=0)
    w, v = np.linalg.eigh(z)
    if w.min() <= 1e-12 * w.max():
        raise ValueError("tomography settings are not informationally complete")
    zih = (v / np.sqrt(w)) @ v.conj().T
    povm = np.einsum("ab,kbc,cd->kad", zih, projectors, zih)
    total = n.sum()
    if total <= 0:
        return np.eye(z.shape[0], dtype=complex) / z.shape[0]
    f = n / total
    sigma = np.eye(z.shape[0], dtype=complex) / z.shape[0]
    for _ in range(max_iter):
        p = np.real(np.einsum("kab,ba->k", povm, sigma))
        ratio = np.where(p > 1e-300, f / np.maximum(p, 1e-300), 0.0)
        r = np.einsum("k,kab->ab", ratio, povm)
        new = r @ sigma @ r
        new = (new + new.conj().T) / 2
        new /= np.real(np.trace(new))
        done = np.max(np.abs(new - sigma)) < tol
        sigma = new
        if done:
            break
    rho = zih @ sigma @ zih
    rho = (rho + rho.conj().T) / 2
    return rho / np.real(np.trace(rho))


def linear_ket(theta):
    return np.array([np.cos(theta), np.sin(theta)], dtype=complex)


def correlation_from_state(rho, ket_a, ket_b):
    """E = <A (x) B> for analysers A = |a><a| - |a_perp><a_perp|, likewise B, on rho (4x4)."""
    r = _as_density(rho)

    def obs(k):
        p = np.outer(k, k.conj())
        return 2 * p - np.eye(2)

    return float(np.real(np.trace(np.kron(obs(ket_a), obs(ket_b)) @ r)))


def correlation_from_angles(rho, theta_a, theta_b):
    """E for linear analysers at angles theta_a, theta_b (rad from H) on the signal and idler."""
    return correlation_from_state(rho, linear_ket(theta_a), linear_ket(theta_b))


CHSH_ANGLES_DEG = (0.0, 45.0, 22.5, 67.5)


def chsh_from_state(rho, angles=None):
    """CHSH S for linear analysers; angles = (a, a', b, b') in rad (default optimal, Phi+)."""
    a, ap, b, bp = np.deg2rad(CHSH_ANGLES_DEG) if angles is None else angles
    e = correlation_from_angles
    return chsh_s(e(rho, a, b), e(rho, a, bp), e(rho, ap, b), e(rho, ap, bp))


def swap_qubits(rho):
    r = _as_density(rho)
    return r.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)


def reduced_states(rho):
    r = _as_density(rho).reshape(2, 2, 2, 2)
    return np.einsum("ijkj->ik", r), np.einsum("jijk->ik", r)


def stokes_of(rho):
    # same convention as Beam.stokes: S3 = -2 Im rho_HV, so R = (H - iV)/sqrt2 has S3 = -1
    r = np.asarray(rho, dtype=complex)
    return np.array(
        [(r[0, 0] + r[1, 1]).real, (r[0, 0] - r[1, 1]).real, 2 * r[0, 1].real, -2 * r[0, 1].imag]
    )


def conditional_state(rho, ket_a):
    """State of qubit b after qubit a passes the analyser |ket_a>, and the probability p."""
    k = np.asarray(ket_a, dtype=complex)
    k = k / np.linalg.norm(k)
    r = _as_density(rho).reshape(2, 2, 2, 2)
    m = np.einsum("i,ijkl,k->jl", k.conj(), r, k)
    p = float(np.real(np.trace(m)))
    if p <= 1e-12 * float(np.real(np.trace(_as_density(rho)))):
        return np.full((2, 2), complex(np.nan, np.nan)), max(p, 0.0)
    return m / p, p


def pair_state_observables(rho):
    r = _as_density(rho)
    ra, rb = reduced_states(r)
    conditional = {}
    for label, ket in KETS.items():
        m, p = conditional_state(r, ket)
        conditional[label] = {"probability": p, "rho_b": m, "stokes_b": stokes_of(m)}
    return {
        "purity": purity(r),
        "concurrence": concurrence(r),
        "bell_fidelity": {name: fidelity(r, psi) for name, psi in BELL.items()},
        "reduced_a": ra,
        "reduced_b": rb,
        "stokes_a": stokes_of(ra),
        "stokes_b": stokes_of(rb),
        "conditional": conditional,
    }


def key_metrics(analyses: dict) -> list[dict]:
    """One row per headline figure of merit in the rate engine's analyses."""
    rows: list[dict] = []

    def add(metric, value, name, definition, unit=""):
        if value is None:
            return
        rows.append(
            {
                "metric": metric,
                "value": value,
                "unit": unit,
                "analysis": name,
                "definition": definition,
            }
        )

    pairs = {}
    for name, a in analyses.items():
        kind = a.get("type")
        if kind == "coincidences":
            w = a.get("window_s", float("nan"))
            add(
                "CAR",
                a.get("car"),
                name,
                f"true / accidental coincidences in a {w * 1e9:.3g} ns window",
            )
            add(
                "heralding efficiency",
                a.get("heralding_b_given_a"),
                name,
                f"true coincidences / singles of {a['channels'][0]}",
            )
            add(
                "heralding efficiency",
                a.get("heralding_a_given_b"),
                name,
                f"true coincidences / singles of {a['channels'][1]}",
            )
            add(
                "two-photon interference visibility",
                a.get("two_photon_visibility"),
                name,
                "1 - C / C_distinguishable (HOM-type; 1 = perfect dip, 0 = no interference)",
            )
            pairs[tuple(a["channels"])] = (name, a.get("true_hz"))
        elif kind == "heralded_g2":
            add(
                "heralded g2(0)",
                a.get("g2_heralded"),
                name,
                "N_hab N_h / (N_ha N_hb); 0 for an ideal single photon, < 0.5 single-photon regime",
            )
        elif kind in ("g2", "histogram"):
            add(
                "unheralded g2(0)",
                a.get("g2_at_center"),
                name,
                "normalised coincidences at zero delay; 1 + 1/K for K Schmidt modes, 2 thermal",
            )
    firsts = sorted({a for a, _ in pairs})
    seconds = sorted({b for _, b in pairs})
    for a1, a2 in [(x, y) for i, x in enumerate(firsts) for y in firsts[i + 1 :]]:
        for b1, b2 in [(x, y) for i, x in enumerate(seconds) for y in seconds[i + 1 :]]:
            keys = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
            if not all(k in pairs for k in keys):
                continue
            c = [pairs[k][1] or 0.0 for k in keys]
            total = sum(c)
            if total <= 0:
                continue
            e = (c[0] - c[1] - c[2] + c[3]) / total
            names = "/".join(pairs[k][0] for k in keys)
            add(
                "correlation E",
                e,
                names,
                "(C++ - C+- - C-+ + C--) / sum at the current analyser settings",
            )
            add(
                "correlation visibility |E|",
                abs(e),
                names,
                "fringe visibility at these settings; maximise over angles (CHSH tab) for V",
            )
    return rows
