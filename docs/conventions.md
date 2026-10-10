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
  using the nominal path delays; the `delay_aliasing` diagnostic fires when a path-delay
  difference reaches half a period (other paths can then alias). For
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
- Accidentals for CW sources use `S_A S_B W` with total singles; CAR is `true / accidental`.
  The heralded g2 uses the uncorrelated-process model `R_Hab = R_Ha S_b W + R_Hb S_a W + S_H S_a S_b W^2`
  (no intrinsic multi-pair term and no cross-talk pairs).
