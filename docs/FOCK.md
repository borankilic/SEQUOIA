# Photon-number (Fock) circuit backend

Optional backend in `sequoia.fock` for experiments that are a linear-optical circuit with
Fock-state inputs (HOM, boson sampling, heralding logic). It is separate from the rate, Monte Carlo
and Gaussian engines and never converts a source: SPDC, multi-pair statistics, spectral
entanglement and timing/jitter are **not** supported. Photons enter as an explicit Fock input.

## Conventions

- Modes are numbered from 0. The transfer matrix `T` maps input amplitudes to output amplitudes,
  `a_out = T a_in`; `T` is contractive (`T†T <= 1`).
- Beam splitter (Perceval Rx): `[[cos t/2, i sin t/2], [i sin t/2, cos t/2]]`, first listed mode is the
  top mode. Optional `phi_tl, phi_bl, phi_tr, phi_br` multiply entries as in Perceval
  (`u00 ~ e^{i(tl+tr)}`, `u01 ~ e^{i(tr+bl)}`, `u10 ~ e^{i(tl+br)}`, `u11 ~ e^{i(bl+br)}`); `Ry` and
  `H` conventions are available.
- Phase shifter: `e^{+i phi}` on the mode. Permutation: the photon in `modes[i]` leaves in
  `modes[perm[i]]`.
- Loss: `Loss(mode, p)` removes a photon with probability `p`. Loss is handled exactly by unitary
  dilation (`T` is stacked on `(1 - T†T)^{1/2}` environment modes and the environment is traced
  out); results are never renormalized. Output states with fewer photons are the loss events, so the
  distribution always sums to 1.
- Distinguishability: each input photon carries a tag. Equal tags interfere, different tags are
  fully incoherent (probability = convolution of the per-tag distributions), as for Perceval's
  `NoisyFockState`. The scalar `indistinguishability` follows Perceval's source model: each photon
  is in the common mode with probability `sqrt(ind)`, otherwise it carries a private tag, so any
  two photons overlap with probability `ind`. `brightness` is an **emission probability per photon**
  (Perceval's meaning), not SEQUOIA's pair brightness in pairs/s/mW; `g2` is not modelled.
- Detectors act on groups of output modes (the summed photon number) and are a response matrix
  `D(k|n)` (`det.response_matrix(n_max)`): `Threshold`, `IdealPNR`, `PseudoPNR(wires=N)` (equal
  multiplexing; `P(k|n) = C(N,k) k! S(n,k) / N^n`, so 2 photons give 2 clicks with probability
  `(N-1)/N`; `PseudoPNR.layered(layers, R)` gives unequal wire weights). `efficiency` is a
  per-photon detection probability and lives **only** on the detector model; the circuit must not
  also carry a `Loss(kind="detector")` on that mode (error). `dark` is an independent dark click
  that adds one count (saturating). The compiler puts SEQUOIA detector efficiency here, not in the
  circuit, unless it is polarization dependent (then its square root is part of the transfer and
  listed in `efficiency_in_circuit`).
- Conditioning: `condition(dist, {mode: value | (op, value)})` returns `success_probability`, the
  unnormalized `joint` and the normalized `probs` separately.

## SEQUOIA setups

`compile_setup_network(setup, library, wavelength, inputs=["pbs.in", ...])` builds the
transfer from an H/V (or `input_basis="DA"|"RL"`) mode pair per input port to an H/V pair per
detector, reusing the components' own Jones transfers and the conventions of `docs/conventions.md`.
Only free-space optics, wave plates, polarizers, filters, fibers, fiber splitters and
single-photon detectors reachable from the inputs are converted; sources and nonlinear crystals
raise `FockCompileError`. Spatial-mode overlap between recombining paths is taken as perfect;
fiber coupling uses `seed_mode`. Path phases `2 pi OPL / lambda` are included at the single given
wavelength; there is no spectral or temporal structure, so a HOM *delay scan* is not available
(use the rate engine).

## Using it

```python
from sequoia.fock import Circuit, Experiment, run_experiment
exp = Experiment(Circuit(2).beam_splitter(0, 1), [1, 1], shots=1000, seed=1)
print(run_experiment(exp).table())
```

Experiment documents are YAML/JSON (`save_experiment`, `load_experiment`); `sequoia.fock.cli.main`
runs one (`--shots --seed --backend --json --csv`; a Perceval GUI circuit needs `--input 1,1,0`).
`import_perceval_gui` reads the Perceval GUI version 1/2 JSON: waveguide loss `db_per_cm * length`
before each element (no phase), active loss after it on both modes, detector efficiency in the
detector model. Optional Perceval (`pip install 'sequoia[perceval]'`): `compare_with_perceval`,
`perceval_distribution`, and `backend: perceval` in an experiment.

## Limits

Cost is dominated by permanents: `~ 2^(n-1) n` per output pattern, patterns grow as
`C(modes + n - 1, n)` (and with environment modes when lossy). `Limits(max_photons=10,
max_work=2e9)` raises `FockResourceError` with an estimate beforehand; 6 photons in 12 modes is
interactive. Cancellation (`should_cancel`) is polled between permanent batches and raises
`sequoia.engines.cache.Cancelled`. Probabilities below 1e-18 are dropped.
