# SEQUOIA conventions

These hold throughout the code. Each one gets a test when the code that depends on it lands.

## Units

- Internal values are SI floats or numpy arrays (m, s, W, Hz, rad).
- Units are parsed with pint only at the boundaries: part files, setup files and GUI input.
- Angles are in degrees in files and the GUI, and in radians internally.
- Wavelengths are vacuum wavelengths.

## Data files

- Scalars are written as `<number> <unit>` (`405 nm`) or, with provenance, as `{value, unit, provenance}`.
  A bare number is dimensionless.
- Curve and surface data are stored in CSV files inside the part folder, in the units named in the
  header. A curve's `x`/`y` fields hold those file units; `x_si`/`y_si` hold SI values.
  Curves and surfaces are always called with SI arguments.
- Axes are written `name (unit)`, for example `wavelength (nm)`. The bracket form `name[unit]` is
  also read, but it needs quoting inside a YAML flow mapping.
- Curves must have strictly increasing x in SI. Queries outside the data range follow the curve's
  `extrapolation` policy: `warn_clamp` (default), `clamp`, `zero` or `error`.
- Wavelength axes imported from frequency, wavenumber or energy are converted to `nm` and sorted.
- Provenance sources: `datasheet`, `raw-data`, `measured`, `model`, `ideal-default`.
- Part references are paths relative to the library root (`detectors/snspd/sq_eos_ch2`).
- Setup links run from an output port to an input port, written `instance.port`. Link kinds are
  `free_space`, `fiber` and `cable`, and each must join ports of the matching kind.
- Fiber links reference a part of type `OpticalFiber`; probes attach to links by id.

## Fields and phases

- Field phase convention: E ∝ exp[i(kz − ωt)].
- Path delays enter the network transfer as phases exp(iωτ).

## Polarization

- Lab frame: H lies in the table plane, V is the table normal, and k is the propagation direction.
- Jones vectors and matrices are written in the (H, V) basis.
- An ideal mirror has Jones matrix diag(1, −1), which flips handedness.
- Coated mirrors add the s–p retardance from coating data. For reflection in the table plane, s = V and p = H.
- Beam splitters are symmetric: r = i√R, t = √T.
- Wave plate, polarizer and analyser angles are positive counter-clockwise when looking against the
  propagation direction (towards the source), i.e. from H towards V in the right-handed (H, V, k)
  frame. The angle is that of the fast axis of a wave plate (the V component is the slow one) or the
  transmission axis of a polarizer, measured from H. The Jones matrix of a plate at angle t is
  `R(t) diag(e^{-iδ/2}, e^{+iδ/2}) R(-t)` with `R(t) = [[cos t, -sin t], [sin t, cos t]]`, so a
  half-wave plate at angle h maps linear polarization at angle θ to 2h − θ.
- Reflection is written in each beam's own right-handed (H, V, k) frame: a mirror or the reflected
  port of a beam splitter is `i·diag(1, −1)` (V changes sign, handedness flips), and transmission is
  `diag(1, 1)`. Consequences for two photons on a symmetric (50:50) beam splitter, one per input port
  and arriving together: HH or VV bunch (HOM dip), ψ+ = (HV + VH)/√2 sends the photons to different
  outputs (all pairs are coincidences), ψ− = (HV − VH)/√2 bunches, and two photons both rotated to
  D = (H + V)/√2 by the same plate do not show a dip, because the reflected D leaves as A and is
  distinguishable from the transmitted D (a dip needs D at one input and A at the other).
  This is the mirror image of the textbook statement, which labels the reflected polarization with
  the same (H, V) as the incident one and so assigns the coincidences to ψ−. It is the physical
  behaviour of a dielectric splitter seen in per-beam frames, tested in
  `tests/test_rates.py::test_symmetric_beam_splitter_sends_psi_plus_to_coincidences_and_psi_minus_to_bunching`.

## Spatial modes

- Optics are paraxial, with one Gaussian transverse mode per path and separate x and y complex beam parameters.
- M² uses the embedded-Gaussian model.
- Alignment is ideal unless misalignment parameters are set.

## Network topology

- Only feed-forward networks are supported: no cavities and no back-reflections.
- A cycle is a validation error.
- Canvas position and rotation are layout only. Optical angles are component parameters.

## Physics simplifications (v1)

- The pump is undepleted and components are stationary.
- Detectors are threshold detectors; PNR arrives with the Gaussian engine (M8).
- The rate engine is first order in the pair rate (low gain).

## Classical engine (M2)

- Every source emits a sampled spectrum. A CW laser is one sample with a Lorentzian linewidth; a
  pulsed laser is 11 samples of a transform-limited Gaussian spectrum. Jones matrices and
  attenuation are evaluated per sample.
- The Gaussian mode is tracked at the source's central wavelength. Lens and fiber data are
  evaluated there, so chromatic focal shift across a broad spectrum is not modelled.
- Paths from one source add coherently on a link, with the optical-path phase per sample, the
  coherence factor of the source spectrum, and the Gaussian-mode overlap between the paths.
  Different sources add incoherently.
- A link reports the mode of its dominant path.
- Lenses and objectives are thin elements at their principal plane. Include the principal-plane
  separation in the link lengths.
- Fiber links apply the overlap with the fiber mode (divided by M²x·M²y), attenuation, connector
  loss, the polarization transform and the phase and group delay. Group-delay dispersion and
  polarization-mode dispersion are not modelled. PM fibers pass polarization unchanged.
- A multimode fiber uses a crude acceptance estimate (core area times NA acceptance).
- Wave plate retardance scales as 1/λ from the design wavelength unless a retardance curve is given.
- Fiber paddles use the bend-birefringence retardance 2π²·a·d²·N/(λ·D) per paddle, with a = 0.133.
- A link's length can be locked to the canvas: the distance between instance positions times the
  grid cell.

## Mixed pair states (ensembles)

A `PairSource` with a member axis is an incoherent mixture of pure pair states. What the engine
computes from it depends on the statistic, and two models are used.

- Linear statistics (singles, coincidences, windows, correlation functions, HOM, Bell correlations)
  are sums over members with their weights and are model independent.
- Pulsed sources, second-order statistics: each pulse is entirely in one member k (probability
  `w_k`). The per pulse g2 is the weighted average `sum_k w_k (1 + 1/K_k)` of the member values (plus
  the covariance of the detected intensities over members), and the heralded bunching is built from
  the same per-member moments. This is the shot-to-shot mixture.
- CW sources, second-order statistics: the field is the Gaussian field defined by the one-body
  density matrix `rho = sum_k w_k rho_k` of the mixture (independent pairs drawn at random from the
  members), g2 follows from its purity.
- `by_role[...]["schmidt_*"]` and `metrics.schmidt_number` are the participation ratio of the
  averaged one-body density matrix, the quantity that controls the CW g2.

For members that share a common reduced state (polarization mixing of one JSA, the Werner state)
the models coincide and `g2 = 1 + 1/K`. For members with different spectral structure they do not:
two members `f1 g1` and `f2 g2` in disjoint bands give g2 = 2 per pulse but a Schmidt number of 2,
for which `1 + 1/K` would be 1.5. The engine reports `ensemble_g2_model` when a g2, heralded g2 or
Schmidt number is requested for such a source. A pure multimode JSA must never be encoded as an
ensemble of its Schmidt modes (the per pulse g2 would be 2, not `1 + 1/K`); give it as one pure
amplitude.

## Run-time behaviour (engine API)

- `simulate(setup, library, *, should_cancel, cache, negligible_power_fraction,
  min_power_for_warnings)` and `simulate_rates(setup, library, classical, *, should_cancel,
  cache, max_memory_bytes)`. `should_cancel` is a callable polled between stages and inside long
  loops; when it returns True the run raises `engines.cache.Cancelled`. Results are never
  partial.
- `RunCache` (module `engines/cache.py`) keeps parsed parts (keyed by folder and the mtimes and
  sizes of its files) and `emit_pairs` results. A component enables emission caching with the
  optional methods `emit_cache_key(lights)` (hashable key of everything the emission depends on
  that is not in the parameters and settings, or None) and `emit_rate_scale(lights)` (a factor
  that the pair rate is linear in, such as the pump power; the cached rate is rescaled when only
  that changed). The cache key always includes the component type, parameters and settings.
  Components without these methods are never cached.
- Warnings of the data layer (curve extrapolation, detector configuration) are captured by a
  thread-local recorder behind a process-wide `warnings.showwarning` hook, so concurrent runs in
  several threads do not see each other's warnings and no global warning state is toggled per run.
- Memory: before allocating, the engine estimates the arrays of the lattice, the time-correlation
  FFT and the second-order moments and raises `ResourceLimitError` (a `SetupError`) naming the
  stage and the estimate in GiB when it exceeds `max_memory_bytes` (default 3 GiB). Inside an
  analysis it becomes an error issue `memory_limit` and the analysis is skipped (an unheralded
  pulsed g2 keeps its histogram and drops the per-pulse number). The per pulse moments are evaluated
  in the smaller of the arm and partner spaces.
- Detector saturation: `detector_saturation` when dead time (detector or tagger) removes more than
  10% of the incident counts of a channel, `leakage_saturation` when classical light alone does,
  both with the numbers. `no_phase_matched_pairs` when fewer than 1e-6 of the pairs of a source are
  detected by any detector (the pair spectrum is outside the passbands, or the crystal is detuned).
- `fiber_multimode` is not reported for light below `negligible_power_fraction` (default 1e-6) of the
  total source power, nor below `min_power_for_warnings` watts.

## Fiber links

- The SMF polarization transform defaults to the identity, which is the ideal case.
- The alternatives are an explicit SU(2) or a seeded random one.

## Quantum layer (M3)

These are the approximations a user must know. Details and validation numbers are in
`docs/M3_REPORT.md`.

- First order in the pair rate: the rate engine gives exact single-pair statistics. Multi-pair
  events enter only through the uncorrelated-process formula for heralded g2 and the thermal term
  of the unheralded g2 histogram. There is no high-gain or PNR physics (M8).
- Photons travel the same linear network as classical light: the pair state is transformed with
  the per-wavelength Jones matrices and the phase `nu*gpl/c + nu0*(opl - gpl)/c`. Fiber group delay
  uses the part's `group_index`; fiber dispersion (GDD) and polarization-mode dispersion are
  not modelled.
- The pair state lives on an integer frequency lattice `nu = n*dnu`. Time structure is periodic
  with period `1/dnu`. The correlation peak is found modulo that period and reported unwrapped
  using the nominal path delays. A single path per detector pair needs no unwrapping care: a long
  fibre (even tens of periods) shifts the peak, nothing aliases, and window fractions and centres are
  exact. Aliasing is a property of several interfering paths: `delay_aliasing` fires only when
  two paths of a detector pair with comparable weight (at least 1e-3 of the strongest) differ in delay
  by more than half a period, since then their peaks fold onto each other. Histograms and
  coincidence windows show one copy of the correlation, clipped to the principal period around
  the peak, so periodic images of the lattice never appear (an explicit centre at an alias of a
  long delay gives zero counts, as in the lab). For
  `SPDCCrystal` the lattice follows `delay_window` (the parts set 40 ns: fiber length differences
  up to about 4 m are safe), limited by `max_samples`.
- The spatial mode of each photon is a single Gaussian (the collection mode `signal_waist`,
  `idler_waist` at the position spdcalc optimises) propagated through the optics at the photon's
  central wavelength. Fiber coupling is the Gaussian overlap with the fiber mode. The pair rate is
  computed for these collection waists and is not recomputed when the real optics back-propagate
  a different fiber mode to the crystal, so choose the waists to match (the `coupling efficiency`
  of the photon is then close to the product of component transmissions). There is no chromatic
  focal shift across the pair spectrum.
- The classical pump (leakage) is coupled into the single-mode fiber with the same mode-overlap
  model as the photons, at the pump wavelength, using the fiber's MFD at its design wavelength.
  It ignores multimode guidance below cutoff and the real chromatic mismatch of the collimators, so
  leakage numbers are an estimate to within an order of magnitude; the engine warns
  `fiber_multimode`.
- `SPDCCrystal` keeps the signal band where |JSA|^2 on the phase-matching ridge exceeds
  `spectral_cutoff` (default 1%) of its peak. This drops the far sinc side lobes: the pair rate of
  the example crystal agrees with spdcalc to 0.1%, but HOM dip shapes near the apex differ by up to
  0.02 (absolute, normalised rate) from the full exchange integral.
- The polarization state and `Werner` visibility of `EntangledPairSource` are an ensemble on
  interleaved sum-frequency entries (exact for singles, coincidences, HOM, CW g2; approximate for
  pulsed unheralded g2 and Schmidt number at visibility < 1).
- Wave plates follow `E ~ exp[i(kz - wt)]`: the Jones matrix of a plate at angle t is
  `R(t) diag(e^{-i d/2}, e^{+i d/2}) R(-t)`, which is `diag(1, -1)` for a half-wave plate and
  `diag(1, i)` (up to a global phase) for a quarter-wave plate. `metrics.tomography_settings()`
  returns HWP/QWP angle pairs for an analyser arm HWP -> QWP -> PBS (H transmitted); a half-wave
  plate at `h` analyses linear polarization at `2h`. R = (H - iV)/sqrt2, L = (H + iV)/sqrt2. This is
  checked against the engine in `tests/test_m3_integration.py`.
- A PBS transmits H (`out_t`) and reflects V (`out_r`) with the usual `i` phase; `PBS` parts with
  finite extinction send leakage photons to the wrong arm and the engine counts them.
- Detector dead time is non-paralyzable and applied to singles and coincidences; afterpulsing adds
  to the background; jitter is a Gaussian plus optional exponential tail convolved into the
  correlation function. Dark counts and leakage add incoherently.
- Accidentals for CW sources use `S_A S_B W` with total singles. For pulsed sources the
  same-pulse term `f p_A p_B` is multiplied by the number of pulse offsets whose arrival-time
  difference falls in the window, computed from the jitter distribution of the difference
  folded onto the repetition period: it is 1 inside one period for sharp timing, reduced by the
  jitter window fraction when W is comparable to the jitter, and a staircase
  `2 floor(W / 2T) + 1` for W beyond the pulse period T. The pump pulse width and the time
  structure of the photon arrival are neglected in this term.
- CAR is `C_true / ACC` (true coincidences in the window over accidentals in the same window; the
  value printed in the report as "CAR = C_true/ACC"). Many labs quote `(C_true + ACC) / ACC =
  C_total / ACC`, which is larger by exactly 1; the analysis dict has it as `car_total` and the report
  prints both. The two differ materially below CAR of about 10. CAR is NaN when there are no
  coincidences and no accidentals, and infinite for true coincidences with zero accidentals.
- Heralded g2 uses the uncorrelated-process model
  `R_Hab = R_Ha S_b W + R_Hb S_a W + S_H S_a S_b W^2` with three additions that matter when the
  source is multimode or the herald is efficient (no intrinsic multi-pair term of the source and no
  cross-talk pairs):
  - Pulsed: two pairs in one pulse, the herald fires for either of the two signal photons, so the
    pair-pair terms carry `(1 - eta_H/2)` and the low-mu value is `mu (2 - eta_H) (1 + B)`, B the
    herald-conditioned bunching `1/K` (exact for a herald efficiency flat over the pair spectrum,
    which is how eta_H enters). `eta_H` is taken per ensemble member as the herald's incident pair
    singles over the member's pair rate.
  - Pulsed: B follows the herald filter. The same-pulse triple moment is built from the filtered
    one-photon operators with the herald weighting, so for a mixture whose members reach the herald
    with different spectral weight (a spectral herald filter) B is the bunching of the pairs that
    actually herald. For a single pure state, or members with a common reduced state, it equals the
    unconditional `1/K`.
  - CW: the thermal bunching of a and b photons of different pairs adds `B_W = (1/(S_a S_b W))
    * integral over the b window of the jitter-convolved |G1|^2 profile` to the pair-pair terms (and
    the two-window integral to the accidental triples); `B_W -> 1` for W much shorter than the
    coherence time and `~tau_c/W` otherwise (about 3% for a THz-wide SPDC source and a 2 ns window).
    The window is centred as the coincidence windows are, and the photon that heralds is assumed
    to arrive at the window centre; against the exact Gaussian-state triple integral this is within
    0.5% of the correction for a Gaussian spectrum at W between 0.3 and 5 coherence times.
