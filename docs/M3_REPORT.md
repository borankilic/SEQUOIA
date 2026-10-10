# M3 report: quantum sources, detectors, rate engine, integration

Status of the M3 integration: three example setups, a seed library, end-to-end tests
(`tests/test_m3_integration.py`, helper `tests/m3_helpers.py`). All numbers below are reproduced by
those tests unless marked otherwise.

## 1. What was built

| Item | Where |
|---|---|
| Seed parts (all generic placeholders, provenance `ideal-default` or `model`) | `library/sources/laser_405_cw`, `nonlinear/{ppktp_type2_10mm,entangled_810_generic}`, `filters/{lp_650_pumpblock,bp_810_10nm}`, `polarization/{hwp_810,qwp_810,pbs_ideal,pbs_cube_810}`, `freespace/lens_f200mm`, `fiber/{coupler_f11mm_810,smf_810,beamsplitter_50_50_810}`, `detectors/{spad_si_generic,snspd_generic}`, `electronics/tagger_generic_4ch` |
| Heralded HBT | `setups/hbt_heralded.yaml` |
| HOM | `setups/hom.yaml` |
| Bell/CHSH/tomography | `setups/bell_test.yaml` |
| Tests | `tests/test_m3_integration.py`, `tests/m3_helpers.py` |

Part conventions: curves are generic (LP 650 nm: OD 8 stop band, 0.95 pass band; BP 10 nm flat-top: OD 6
stop band, 0.93 peak; Si SPAD PDE 0.55 at 810 nm; SNSPD SDE 0.85 at 810 nm). The SPAD has 200 Hz dark
counts, 50 ns dead time, 350 ps jitter. The SNSPD has 100 Hz, 20 ns, 30 ps and polarization
extinction 1. The crystal's `d_eff` is the bulk d24 literature value (3.9 pm/V), not measured.
`delay_window` is 40 ns and `max_samples` 100000 in the crystal part (see 4).

Setup designs:

- `hbt_heralded`: 405 nm, 10 mW, w0 = 0.4 mm, f = 200 mm lens focusing to w = 64 um at the crystal;
  the collection lens (f = 200 mm, 197 mm from the crystal exit face) collimates the 50 um collection
  mode, so the 11 mm fiber collimators couple it with 99.1% overlap. LP + BP -> cube PBS (H = signal
  to the beam splitter, V = idler to the herald). Fiber lengths: herald 1 m, signal 2 m + 1 m.
- `hom`: same source optics, ideal PBS, HWP at 45 deg in the idler arm, link `delay_arm` is the
  adjustable delay, crystal held at 27.4 degC (degenerate; at 25 degC signal and idler spectra are
  centred 0.11 nm apart).
- `bell_test`: `EntangledPairSource` (phi+, V = 0.97 via override), per arm HWP -> QWP -> PBS ->
  both outputs to SNSPDs, so all four outcomes are recorded.

## 2. Physics model as implemented

Pair state (`SPDCCrystal`): spdcalc JSA along the phase-matching ridge for a CW pump (one sum-frequency
entry on the lattice `nu = n dnu`) or over a pump-bandwidth window (finite linewidth / pulsed).
Polarization labels come from the PM type (`Type2_e_eo`: signal H, idler V). Absolute rate: spdcalc's
coincidence rate for the pump waist at the crystal, the collection waists and the incident pump power
(linear in power and `d_eff^2`).

Transport: signal and idler photons are launched as classical beams of the photon's wavelength set
and collection mode through the same network (`propagate_network`), giving a per-detector Jones
transfer with phase `nu*gpl/c + nu0*(opl - gpl)/c`; detector efficiency operators multiply on top.

Rates: coincidences `R * sum psi^dagger (E_A (x) E_B) psi` with the exchange term (HOM), singles
from the reduced polarization density matrices, dark counts + afterpulsing + classical leakage,
non-paralyzable dead time, correlation function by zero-padded FFT convolved with detector and
tagger jitter, accidentals `S_A S_B W`, heralded g2 from the uncorrelated-process formula.

## 3. Validation results

### 3.1 HBT (`test_hbt_*`)

Hand calculation from part data: spectral averages of LP x BP x PDE over the actual signal/idler
marginals, PBS T/R values, collimator transmission, fiber attenuation, beam splitter, and the
Gaussian-mode overlap computed with my own ABCD/overlap code (independent of `light/gaussian.py`) from
the apparent collection waist (waist position from spdcalc's `auto`).

| Quantity | Engine | Hand calculation |
|---|---|---|
| pair rate R (10 mW) | 2.5808e5 /s | 2.5839e5 /s (spdcalc `efficiencies`, ratio 0.9988) |
| herald pair singles (before dead time) | 1.2660e5 /s | 1.2660e5 /s |
| signal-arm detector pair singles | 5.837e4 /s | 5.837e4 /s |
| herald singles after dead time | 1.2601e5 /s | 1.2601e5 /s |
| all-time coincidence herald x a | 2.6982e4 /s | 2.6982e4 /s |
| window fraction (1 ns, 350 ps SPAD + 10 ps tagger jitter) | 0.98258 | 0.98258 (erf) |
| true coincidences | 2.6511e4 /s | 2.6512e4 /s |
| accidentals (`S_h S_a W`) | 7.362 /s | 7.362 /s |
| CAR | 3601 | 3601 |
| heralding efficiency, signal arm (a given herald) | 0.2104 | 0.2104 |
| heralded g2(0) (window 1 ns) | 5.6e-4 | formula, 1e-9 relative |
| classical leakage herald / signal arm | 2.84 / 26.7 /s | 2.84 / 26.7 /s |

The agreement is at the 1e-6 level because the hand calculation uses the same physical model with
independently written code; it validates the wiring (parts -> network -> detectors -> analyses),
not the model assumptions. Tolerances in the tests are 0.2-3 percent.

Coincidence peak position: -9.791 ns versus -n_g * 2 m / c = -9.793 ns (difference 2 ps, the
crystal's own signal-idler walk-off). The lattice period is 40 ns, so no aliasing; extra fiber of
1 m and 2 m is tracked linearly up to 4 m total difference (19.6 ns); 7 m triggers `delay_aliasing`.

Pump leakage: nominally 2.8 Hz, far below 1% of singles (no diagnostic). Without the LP filter
(instance removed) the engine reports `pump_leakage` at all three detectors with 2.84e8 /s (herald)
and 2.67e9 /s (signal arm) incident counts, equal to the hand value to 3% and a factor 1e8 = 1/T_LP
above nominal; the detectors saturate near 1/dead time = 2e7 /s (pair singles 8.8e3 /s at
the herald). That is the expected blinding of a Si SPAD by a few mW of 405 nm light behind a single
OD 6 band-pass.

### 3.2 HOM (`test_hom_*`)

The dip is centred at 2.25 ps extra signal-arm path (crystal walk-off; the same offset in
spdcalc). Normalised coincidence rate versus delay:

| Comparison | Max abs difference | Dip floor (engine / reference) | FWHM (engine / reference) |
|---|---|---|---|
| CW engine vs spdcalc `jsa()` exchange integral, 25 degC | 0.020 | 0.0477 / 0.0478 | 2.052 / 2.058 ps |
| CW engine vs same, 27.4 degC (degenerate) | 0.020 | 1.1e-4 / 0.9e-4 | 2.015 / 2.017 ps |
| finite pump 0.0405 nm vs `hom_rate_series` | 0.0019 | 0.372 / 0.370 | 2.399 / 2.402 ps |
| finite pump 0.01 nm vs `hom_rate_series` | 0.0008 | 0.151 / 0.152 | 2.087 / 2.088 ps |

Normalisation: each curve divided by the mean of the points at least 4.5 ps from the dip centre.
The engine's finite-pump runs use a Gaussian pump spectrum (laser `pulse_width` override), the
spdcalc reference uses `hom_rate_series` with `optimum_range(300)`; 11 pump samples enter the engine.
Tolerances in the tests are 0.01 for the finite-pump series and 0.025 for the CW series. The CW series is
compared to the direct spdcalc `jsa()` exchange integral because `hom_rate_series` itself is not
converged for pump bandwidths below 0.01 nm (its dip floor keeps falling and the baseline ripples by
1 percent, with a result that depends on `optimum_range`), so it is no reference for the CW limit. The residual 0.02
is at the dip shoulders (+-0.25 ps from the centre), caused by the 1% spectral cutoff of the
crystal's grid; the dip floor itself agrees to 1e-4.

Physical checks: the dip visibility is 0.95 at 25 degC (signal and idler centred 0.054 nm either
side of 810 nm), 0.9999 at the degenerate 27.4 degC; rotating the idler HWP to 0 deg
(distinguishable photons) gives a flat curve to 1e-3; the dip depth decreases monotonically
0.37 -> 0.15 -> 0.05 for pump bandwidth 0.0405 nm, 0.01 nm and CW.

### 3.3 Bell test (`test_bell_*`, `test_tomography_*`)

- Absolute coincidence rate of the example (a = 0, b = 22.5 deg) equals
  `R eta_s eta_i (1 + V cos 45 deg)/4` with eta from the part data to 0.3%.
- E(a, 22.5 deg) for a = 0 ... 157.5 deg follows `V cos 2(a - b)` to 2e-3.
- CHSH (0, 45, 22.5, 67.5 deg) with the real HWP + PBS components gives S = 2.74357 versus
  `2 sqrt(2) V = 2.74357` (V = 0.97); with accidentals included S is lower by less than 0.005.
  V = 1 gives 2.8284; V = 0.5 gives 1.4142 (2e-3).
- Polarization conventions: the 16 `metrics.tomography_settings()` angle pairs, applied to the real
  `WavePlate` components, give rates proportional to `tr(P_k rho)` for rho = V|psi><psi| + (1-V) I/4 with
  psi = (|HH> + e^{i phi}|VV>)/sqrt2, phi = 0, +90, -60 deg, V = 0.92 (residual below 2e-3 of the scale).
  `mle_density_matrix` of the noiseless engine rates reproduces the state with fidelity > 0.9995
  (max element error 5e-3); the complex-conjugate state (flipped R/L) would have fidelity < 0.8 for
  the phase = 90 deg and -60 deg cases, so the test distinguishes the sign convention. With Poisson
  noise (rate scale x 1 s) the fidelity is > 0.99.

### 3.4 Lattice / delays (`test_hbt_fiber_*`, `test_hbt_with_short_lattice_*`)

With the library crystal (`delay_window` 40 ns, `dnu` = 25 MHz, 37 000 lattice points, about 1.5 s to
build) the 2 m fiber imbalance (9.79 ns) is far inside the unambiguous range of +-20 ns, and the
`delay_aliasing` diagnostic fires for none of the three examples (asserted). Added fiber length
(1 m, 2 m extra; total imbalance up to 19.6 ns) moves the reported peak linearly, window fraction
unchanged to 1e-6. With the crystal's default grid (`delay_window` 10 ns, period 10 ns) the same
2 m imbalance exceeds half a period: the engine raises `delay_aliasing`, reports the peak
unwrapped (-9.79 ns, from the nominal path delays) and the counts and CAR are unchanged to 1e-3.
7 m of fiber (34 ns > period/2 = 20 ns) is flagged the same way.

## 4. Defects found and changes made in `src/`

1. `src/sequoia/engines/rates.py`, `check_delays`: the `delay_aliasing` diagnostic only compared the
   spread of path-delay differences, so a single large offset (e.g. 2 m of fiber in one arm with a
   10 ns lattice period) passed silently and the correlation peak came out modulo the period.
   I added a second branch that warns when the largest absolute path-delay difference exceeds half
   the period. While I was integrating, the same file was extended concurrently (not by me) with
   unwrapping of the reported peak from the nominal path delays and `tests/test_rates.py` cases for
   it; my branch is still in place and the final message/behaviour there is theirs. My integration
   tests assert the combined behaviour.
2. Not a code change: the grid choice. The crystal's defaults (`delay_window` 10 ns,
   `max_samples` 20000) alias for imbalances above 5 ns; the `ppktp_type2_10mm` part therefore sets
   `delay_window: 40 ns` and `max_samples: 100000`. No change to `nonlinear.py` was needed for this.
3. Observed, not fixed: spdcalc's `hom_rate_series` is not converged for pump bandwidths below
   about 0.01 nm (it is not a valid CW-limit reference); see 3.2.

## 5. Known limitations and open points

- The pair rate depends on the collection waists in the part, not on the optics actually placed
  after the crystal; no back-propagation of the fiber mode is implemented (PLAN M3 item). The
  HBT example is built so the two agree (collimator overlap 99.1%).
- Pump leakage uses the same mode-overlap model at 405 nm as at 810 nm and the single-mode fiber's
  MFD; real leakage depends on chromatic mismatch and below-cutoff guiding. It is order-of-magnitude.
- `SPDCCrystal` cuts the JSA at 1% of its peak along the ridge: HOM dip shoulders are good to 0.02
  (absolute), the dip floor to 1e-3; lower `spectral_cutoff` costs lattice points.
- The collection-mode apparent waist position uses the mean of the three principal refractive
  indices. The effect on the waist position is below 0.02 mm.
- `d_eff` is a literature value; the absolute brightness is not validated against a measurement
  (planned, PLAN section 8). Use the part's `brightness` field for a measured number.
- Heralded g2 is the uncorrelated-process estimate: no intrinsic multi-pair term of the SPDC
  source and no cross-talk pairs from a finite-extinction PBS (27 Hz of true a x b coincidences in
  the HBT example, from the T_s = 0.001 leakage of the idler into the signal arm, visible in the
  `g2_ab` histogram as an apparent g2 peak of 15).
- Fiber polarization is the identity (ideal); no paddles in the examples.
- The engine prints duplicate `fiber_multimode` diagnostics for the 405 nm pump (classical and
  photon pass); cosmetic.
- One setup run builds the crystal once (about 1.5-3 s); the spdcalc evaluation (about 50 us per
  JSA point) dominates. The integration tests cache it.
