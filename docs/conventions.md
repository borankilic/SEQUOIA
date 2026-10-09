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
