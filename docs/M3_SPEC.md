# M3 specification: quantum sources, detectors, rate engine

Read `docs/PLAN.md` (sections 2.2, 3, 7-M3, 8), `docs/conventions.md` and the existing code in
`src/sequoia/` first. M0-M2 are done and green (`uv run pytest`). The interface layer below already
exists; do not change it without telling the supervisor.

## Ground rules for every agent

- Touch only the files assigned to you. Other agents are editing other files in the same tree at the
  same time. Never run `git add`, `git commit`, `git checkout` or `git stash`, and never run
  `ruff format .` over the whole repo (format only your own files).
- No comments or docstrings in code unless they explain something non-obvious. Match the existing
  code style (see `src/sequoia/components/freespace.py`, `src/sequoia/engines/classical.py`).
- Tests go in your own `tests/test_*.py` files. The full suite may show failures caused by other
  agents' unfinished work; run your own tests, and report anything else you saw.
- Environment: `uv run pytest`, `uv run ruff check <files>`. The venv lives in `venv/` (symlinked as
  `.venv`); if `import sequoia` fails, run `chflags nohidden venv/lib/python3.12/site-packages/*.pth`.
- Physics must be checked against analytic results, not only against your own code. When a number
  matters, derive the expected value independently in the test.
- Report: what you built, what you verified (with numbers), what you could not do or are unsure of.

## Existing interface layer (already written)

- `light/pairs.py`: `PairSource`, `PairEmission`. Frequencies live on an integer lattice
  `nu = n * dnu`; `amplitude[sigma_s, sigma_i, j, m]` with signal frequency `n_signal[j]*dnu`, idler
  frequency `(n_sum[m] - n_signal[j])*dnu`, normalised `sum |amplitude|^2 = 1`; `rate` = pairs/s.
  CW pump: `len(n_sum) == 1` (or a few entries for a finite pump linewidth). Pulsed: many `n_sum`.
  Polarization index 0 = H, 1 = V.
- `light/detection.py`: `DetectorModel` (efficiency operator `(N,2,2)` per wavelength, dark counts,
  dead time, jitter with optional exponential tail, afterpulsing), `TaggerModel`.
- `components/base.py`: `Component.emit_pairs(lights, ctx) -> PairEmission | None`,
  `Component.detector_model()`, `Component.tagger_model()`, plus the existing `propagate`, `detect`.
- `light/state.py`: `Beam` (per-wavelength Jones `(N,2,2)`, `opl`, `gpl`, `mode`, `offset`),
  `LinkLight`, `combine`. `engines/classical.py`: `simulate()`.
- `setup/model.py`: cables are `CableLink` (kind `cable`, `delay`). `analyses` is a list of free
  dicts that the rate engine interprets (below).

## Physics (the contract between the work packages)

### Pair state and network transfer

Emitted state: `|psi> = sum F[s,i](nu_s,nu_i) a+_s(nu_s) b+_i(nu_i) |0>`, where `a` is the signal
mode at `signal_port` and `b` the idler mode at `idler_port` (they may be the same port: a collinear
source; then roles are separated downstream by polarization/wavelength optics).

The photons propagate through the same linear network as classical light. The engine launches two
`Beam`s, role `signal` (wavelengths `c/nu_signal`, mode `signal_mode`) and role `idler` (idler
lattice frequencies, mode `idler_mode`), through `Component.propagate`. For a detector `D` and role
`r` the transfer is the coherent sum over all paths `p` of role `r` that arrive at `D`:

`T[D,r](nu) = sum_p J_p(nu) * exp(i*phi_p(nu))`, `phi_p(nu) = 2*pi*( nu*gpl_p/c + nu0_r*(opl_p - gpl_p)/c )`

with `nu0_r` the central frequency of the role (this puts the group delay on the envelope and the
phase delay on the carrier). `J_p` is the beam's `(N,2,2)` Jones array (it already contains losses,
fiber coupling `sqrt(eta)` and polarization transforms).

### Rates (first order in the pair rate)

Detector operators `E_D(nu)` come from `DetectorModel.efficiency`. For an ordered detector pair
`(A, B)` on frequency pair `(nu_A, nu_B)`, with `pq` the final polarization indices:

```
psi_pq(nu_A,nu_B) = sum_{s,i} T[A,sig](nu_A)_{p s} T[B,idl](nu_B)_{q i} F_{s i}(nu_A, nu_B)
                  + sum_{s,i} T[A,idl](nu_A)_{p i} T[B,sig](nu_B)_{q s} F_{s i}(nu_B, nu_A)
coincidence rate (all times) C_AB = R * sum_{nu_A,nu_B} psi^dagger (E_A (x) E_B) psi
```

(a term is zero where the frequency is not on the corresponding lattice; both assignments add
coherently because they lead to the same detection event: this is what produces HOM interference).

Singles from pairs: for role `r` reaching `D`, `S_D^pair = R * sum_nu tr[ E_D T rho_r(nu) T^dagger ]`
with `rho_r` the reduced polarization density matrix at frequency nu of that role (the partner
photon traced out). Roles add incoherently. Add classical leakage (photon rate of the classical
light arriving at the detector port, weighted by `E_D`) and dark counts.

Dead time (non-paralyzable): measured rate = `S / (1 + S * tau_dead)`; coincidences are scaled by
the survival factor `1/(1+S_A tau_A)` of each detector.

### Time structure and coincidence window

At fixed total frequency `nu_A + nu_B = nu_sum` the correlation function of the arrival-time
difference is

`G(tau) = sum_{nu_sum} sum_pq | sum_{nu_A} psi_pq(nu_A, nu_sum - nu_A) exp(-2*pi*i*nu_A*tau) |^2`

(computed with a zero-padded FFT; period `1/dnu`, so choose `dnu` such that this exceeds the
relevant delays). Convolve `G` with the detector jitter of A and B (reflected) and the tagger jitter,
then the coincidence fraction in a window `W` centred on `tau0` is `f_W = int_window G / int G`.
The centre is the peak of the convolved `G` (`center: auto`) or an explicit delay that already
includes cable delays and tagger input delays. Multiple paths give multiple peaks automatically.

Windowed true coincidence rate: `C_true = C_AB * f_W * dead-time factors`.

### Accidentals, CAR, heralding

Split singles into `S^pair` (correlated source) and `S^bg` (darks + leakage).
CW: `ACC = S_A S_B W` (using total singles, i.e. the standard estimate). Pulsed with `W <` pulse
period: `ACC = f_rep p_A p_B + S_A^pair S_B^bg W + S_A^bg S_B^pair W + S_A^bg S_B^bg W`, where
`p = S^pair / f_rep`. `CAR = C_true / ACC`, measured total = `C_true + ACC`.
Heralding (Klyshko) efficiency of A given B: `C_true / S_B`.

Heralded triples (herald H, detectors a, b), uncorrelated-process model:
`R_Hab = R_Ha S_b' W + R_Hb S_a' W + S_H S_a S_b W^2`, `g2_H(0) = R_Hab * S_H / (R_Ha * R_Hb)`.

### Unheralded g2 and Schmidt number

For detectors `a`, `b` fed by the same role `r`, the reduced one-photon operator
`N_ab = E_a^(1/2) T_a rho_r T_b^dagger E_b^(1/2)` (matrix over (polarization, frequency)) gives, for
thermal-statistics SPDC light, `g2_ab = 1 + Tr(N_ab N_ab^dagger) / (Tr N_aa * Tr N_bb)` per pulse.
Schmidt number `K = 1 / sum(lambda_n^2)` of the filtered amplitude; `g2(0) = 1 + 1/K` for a single
pulsed arm. For CW the histogram is `S_a S_b (1 + |G1_ab(tau)|^2 / (S_a S_b))` convolved with the
jitter, `G1_ab(tau) = sum_nu exp(-2 pi i nu tau) tr[N_ab(nu)]`.

### analyses (setup file, interpreted by the rate engine)

```yaml
analyses:
  - {name: ha, type: coincidences, channels: [tagger.ch1, tagger.ch2], window: 1 ns, center: auto}
  - {name: hbt, type: heralded_g2, herald: tagger.ch1, a: tagger.ch2, b: tagger.ch3, window: 1 ns}
  - {name: corr, type: g2, channels: [tagger.ch2, tagger.ch3], range: 5 ns, bins: 200}
  - {name: hist, type: histogram, channels: [tagger.ch1, tagger.ch2], range: 5 ns, bins: 200}
```

Singles for every tagger channel are always reported. Channel names are `instance.port` of the
tagger input ports. Windows and ranges use the `Scalar` shorthand (`1 ns`).

## Work packages

### WP-A: pair sources (`src/sequoia/components/nonlinear.py`, `tests/test_nonlinear.py`)

Implement and register (category `nonlinear`) in `components/nonlinear.py`; import the module from
`components/builtin.py`.

1. `SPDCCrystal`: ports `in` (free-space, IN, required), `out` (free-space, OUT). Params: crystal
   material (spdcalc `crystal_kind`), phase-matching type, length, poling period / theta, d_eff,
   temperature, signal and idler collection-mode waists, optional measured `brightness`
   (pairs/s/mW, overrides the computed rate), spectral grid controls. Settings: `temperature`.
   `propagate` passes the pump through (classical residual pump with absorption loss, jones =
   identity times sqrt(transmission)); `emit_pairs` reads the pump `LinkLight` at `in` (power,
   wavelength, mode, polarization) and builds a `PairSource` from spdcalc (JSA over the phase
   matched band on the integer lattice, polarization index set by the PM type, e.g. type-II:
   signal H, idler V). Pump waist in the crystal comes from the incident pump mode (use the
   beam's `w` at the crystal), pump spectral width from the laser (CW: single `n_sum`).
   Rate: measured brightness x pump power if given; otherwise from spdcalc's rate functions
   (document units and verify scaling: rate is linear in pump power at low gain).
2. `EntangledPairSource`: black-box source with its own pump (no input port). Ports `signal`,
   `idler` (free-space OUT), params: pair rate or brightness x pump power, centre wavelength,
   bandwidth (Gaussian JSA, optionally pulsed), Bell state (`phi+`, `phi-`, `psi+`, `psi-`) and
   visibility/mixing (Werner-type: weight `v` on the Bell state, `(1-v)/4` identity; represent the
   mixed state as an ensemble if needed, or document an approximation and flag it).
3. Must be physically consistent with the engine contract above (normalisation, lattice).
   Provide a helper to build a Gaussian JSA `PairSource` for tests (used by other packages).

Acceptance: unit tests for normalisation and lattice; rate linear in pump power; temperature
tuning shifts the degenerate wavelength in the direction spdcalc predicts; type-II JSA polarization
labels; the JSA agrees with `spdcalc.JointSpectrum.jsa_normalized_range` on the same grid
(compare magnitudes); Bell-state amplitudes exact; visibility mixing correct.

### WP-B: detectors and time tagger (`components/detectors.py` additions, `components/electronics.py`, `tests/test_detectors_m3.py`)

1. `SPAD` and `SNSPD` registered in category `detectors` (add to `components/detectors.py`; keep
   `Photodiode` untouched). Ports: `in` (free-space or fiber, chosen by a `Settings.input`
   `Literal["fiber","free_space"]` defaulting to `fiber`, required) and `out` (electrical OUT,
   required). Params: `efficiency` as `Scalar | Curve` (PDE vs wavelength), dark count rate, dead
   time, jitter FWHM, optional exponential tail (fraction, tau), afterpulse probability and tau,
   active diameter (free-space only). SNSPD adds `polarization_extinction` (ratio of max to min
   efficiency over polarizations) and `polarization_axis` angle; the efficiency operator is
   `eta(lam) * R(theta) diag(1, 1/ext) R(-theta)` scaled so the maximum eigenvalue is `eta(lam)`.
   SNSPD also accepts `efficiency_vs_bias: Surface | None` with setting `bias_current`.
   `detector_model()` returns the `DetectorModel`.
2. `TimeTagger` in `components/electronics.py` (category `electronics`; add the module to
   `components/builtin.py`). Params: `channels` (int, default 4), `jitter_fwhm`, `dead_time`.
   Ports `ch1..chN` (electrical IN, not required). Settings: `input_delays: dict[str, Duration]`
   keyed by port name. `tagger_model()` returns the `TaggerModel`.
3. Photon-flux / saturation helper in `light/detection.py` is out of scope; do not change the
   dataclasses there without agreement.

Acceptance: efficiency operators are Hermitian PSD with max eigenvalue = eta; SNSPD polarization
dependence follows `cos^2`/`sin^2` law versus input polarization angle; curve extrapolation warns
through the existing mechanism; port lists change with `channels`; setup validation accepts a
detector -> cable -> tagger chain and rejects a detector whose `out` is unconnected.

### WP-C: rate engine (`src/sequoia/engines/rates.py`, edits to `engines/classical.py`, `engines/report.py`, `cli.py`; `tests/test_rates.py`)

1. Refactor `engines/classical.py` so the topological propagation core is reusable for photon
   beams (a function that takes seed beams at ports and returns the beams arriving at every
   instance port and link) without changing `simulate()` behaviour (all existing tests must stay
   green).
2. `engines/rates.py`: `simulate_rates(setup, library) -> RateResult` implementing the physics
   contract above: classical pass, `emit_pairs`, photon propagation, per-channel singles, the
   `analyses` list, dead time, diagnostics (`pump_leakage` when classical light reaches a
   detector and produces more than 1% of that channel's singles; `coincidence_outside_window`; no
   tagger connection). Performance: a CW source with 5000 signal samples and 1 sum frequency, two
   detectors, must run well under a few seconds; a pulsed 200x100 JSA also.
3. `RateResult`: `channels[name] -> {singles_hz, pair_singles_hz, background_hz, ...}`,
   `analyses[name] -> dict`, `issues`, plus the classical `SimulationResult`.
4. Extend `engines/report.py` and the CLI (`sequoia run` prints a rates section when the setup has
   photon sources/detectors).
5. Tests must build their own sources/detectors from `PairSource` and `DetectorModel` using small
   test components (do not depend on WP-A/WP-B being finished); verify against closed-form results:
   coincidence rate `R * eta_s * eta_i` for a flat JSA through lossy paths, singles, accidental
   formula, window fraction for a Gaussian biphoton with Gaussian jitter (analytic erf), HOM dip
   visibility 1 and width against the analytic Gaussian JSA result, dead-time formula, polarization
   Malus behaviour for a product state, Bell state correlations, interferometer fringes in one arm.

### WP-D: analysis metrics (`src/sequoia/analysis/metrics.py`, `tests/test_metrics.py`)

Pure functions (numpy only, no engine dependency): `car`, `heralding_efficiency`, `visibility`,
`chsh_s` (from the four correlation values or the 16 rates), `schmidt_number(amplitude)` (SVD of
the (polarization x frequency) matrix), `g2_heralded`, `fidelity`, `purity`, `concurrence`,
`tomography_settings()` (the 16 two-qubit analyser settings as HWP/QWP angle pairs plus the
corresponding projectors), `tomography_counts(rho, settings, rates_total, time, rng)` (Poisson
noise), `mle_density_matrix(counts, settings)` (iterative R-rho-R with convergence check).
Acceptance: Bell state gives S = 2*sqrt(2)*V, concurrence 1; MLE recovers a known Werner state
within statistical error and reduces to the true state with noiseless counts; Schmidt number of
a product state is 1 and of a maximally entangled state is `d`.

## Integration (supervisor, after WP-A..D)

Example setups `setups/hbt_heralded.yaml`, `setups/bell_test.yaml`, `setups/hom.yaml`, seed parts
in `library/`, end-to-end tests, docs.
