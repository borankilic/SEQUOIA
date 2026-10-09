# SEQUOIA conventions

These hold throughout the code. Each one gets a test when the code that depends on it lands.

## Units

- Internal values are SI floats or numpy arrays (m, s, W, Hz, rad).
- Units are parsed with pint only at the boundaries: part files, setup files and GUI input.
- Angles are in degrees in files and the GUI, and in radians internally.
- Wavelengths are vacuum wavelengths.

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

## Fiber links

- The SMF polarization transform defaults to the identity, which is the ideal case.
- The alternatives are an explicit SU(2) or a seeded random one.
