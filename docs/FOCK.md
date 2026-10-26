# Photon-number (Fock) circuit backend

Optional backend in `sequoia.fock` for experiments that are a linear-optical circuit with
Fock-state inputs (HOM, boson sampling, heralding logic). It is separate from the rate, Monte Carlo
and Gaussian engines. The circuit engine itself does not convert a source: SPDC rates, spectral
entanglement and timing/jitter are not modelled by it; the setup integration (see "SEQUOIA
setups") derives pair-number statistics and polarization states from the rate engine. Photons
enter as an explicit Fock input, a coherent superposition, or a statistical mixture of those (see
"Input model").

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
  also carry a `Loss(kind="detector")` on that mode (error). `dark` is the probability of at
  least one dark click on the detector in the window. Threshold and ideal PNR add one count with
  that probability; pseudo-PNR splits it into independent per-wire dark clicks of probability
  `q = 1 - (1 - dark)^(1/N)` (N wires), so k wires holding a detected photon read
  `k + Binomial(N - k, q)`. The compiler puts SEQUOIA detector efficiency here, not in the
  circuit, unless it is polarization dependent (then its square root is part of the transfer and
  listed in `efficiency_in_circuit`).
  The input cutoff `n_max` of `response_matrix(n_max)` only sets the rows; the columns are the
  detector's own outputs: ideal PNR reads up to `n_max + 1` with a dark click, pseudo-PNR always
  has `wires + 1` outputs (dark clicks only on wires that hold no photon), threshold two. So dark
  counts survive on vacuum, one photon plus a dark click reads 2 on a PNR, and vacuum on a
  pseudo-PNR can read several dark clicks.
- Conditioning: `condition(dist, {mode: value | (op, value)})` returns `success_probability`, the
  unnormalized `joint` and the normalized `probs` separately.

## Input model (`sequoia.fock.inputs`)

- `Term(amplitude, photons)`: one Fock configuration, `photons = ((mode, tag), ...)`. Equal tags
  interfere, different tags are distinguishable species.
- `PureState(terms)`: coherent superposition; identical configurations are merged; every term must
  hold the same photon number in every tag class (else `FockError`). The engine normalises it.
- `InputMixture(branches=[(weight, PureState)], notes)`: classical mixture; `omitted_weight = 1 -
  sum(weights)` (e.g. truncated pair numbers) is carried to `Distribution.omitted_weight`, so
  `mass + omitted_weight = 1`.
- `bell_pair(s_h, s_v, i_h, i_v, phase, visibility, kind='phi'|'psi', tag)`: `(|HH> +
  e^{i phase}|VV>)/sqrt2` or `(|HV> + e^{i phase}|VH>)/sqrt2` (signal first). Visibility is the
  Werner form `V|B><B| + (1 - V) I/4`, `I/4` being the uniform mixture of the four product states
  `|HH>,|HV>,|VH>,|VV>` (weights `V` and `(1 - V)/4`). `product_pair(s, i, tag)` is `|s i>`.
- `pair_number_mixture(builder, mean_pairs, schmidt_number=1, n_max=4, statistics='auto')`:
  `builder(tag)` returns the one-pair state with all photons in class `tag`. Pair number:
  `thermal` (`auto`) is negative binomial with mean `mu` and `K` modes,
  `P(n) = C(n+K-1,n) (mu/K)^n (1+mu/K)^-(K+n)` (`K=1`: `mu^n/(1+mu)^(n+1)`; unheralded `g2 = 1 +
  1/K`); `poisson` is `e^-mu mu^n/n!`. Given `n`, the pairs are distributed over the `K`
  equal-weight Schmidt modes (Bose-Einstein = uniform over occupations for thermal, multinomial for
  Poisson); each Schmidt mode is one tag class shared by the partners of a pair and by all pairs in
  that mode, so pairs in the same mode bunch (`K=1`: everything interferes). Occupation patterns
  that differ only by relabelling are merged. `omitted_weight` is the tail beyond `n_max`.

Approximations: non-integer `K` is rounded for tag assignment while the pair-number weights keep
the exact `K` (a note is recorded); a mixed single-pair state (visibility < 1) is combined pair by
pair as independent branch choices; all Schmidt modes have equal weight; no spectral or temporal
structure. Perceval cannot superpose annotated (multi-tag) states, so the Perceval backend accepts
multi-term branches only with one tag class (else `FockError`); the native engine has no such limit.

Engine (`output_distribution_mixture(circuit_or_T, mixture, limits, should_cancel)`): per
branch and per tag class the amplitude is `perm(U[o, c]) / sqrt(prod o! prod c!)` over system and
environment output modes; the term amplitudes are added coherently for every species-resolved
output pattern, squared and summed over splits and environment modes (loss). Single-term branches
use the original `output_distribution` path unchanged. Work grows as the product of the per-class
pattern counts, so many photons spread over many tag classes hit `Limits.max_work` first.

## Counts

Probabilities are per detection window. `expected_counts(dist | conditional, rate_hz, window_s,
duration_s, acquisition)`: windows per second is `rate_hz` (repetition rate), else `1/window_s`;
per second = probability x windows per second; total over `duration_s`, i.e. `N = windows per
second x duration` windows. `sigma` is the predicted spread of the count. The default acquisition
`fixed_windows` treats the N windows as independent trials, so counts are multinomial: `Var N_i =
N p_i (1 - p_i)`, `Cov(N_i, N_j) = -N p_i p_j` (`ExpectedCounts.covariance`, `sigma_sum` for a
sum of outcomes); `acquisition="poisson"` is the rare-event approximation `Var N_i = N p_i`. For a
`Conditional` the unnormalised joint is used, giving the accepted-event rate. `outcome_table(dist,
detectors, top, n_windows, acquisition)` uses the same rule. An `Experiment` takes
`input_mixture`, `rate_hz`, `window_s`, `duration_s`, `acquisition` (document keys
`input_mixture` and `counts: {rate_hz, window_s, duration_s, acquisition}`); `run_experiment` then
fills `FockResult.counts`. Rates and windows must be finite and positive; a duration must be finite
and `>= 0` (zero gives zero counts); a missing duration (None) means 1 s. Invalid values are
errors in the API, the document loader and the GUI alike. Old documents load as before.

Sampling (`sample(dist, shots, seed, tail=...)`): probability the calculation did not resolve
(`1 - mass`, the truncated input weight) is never silently renormalised away. The default
`tail="unresolved"` draws it as an explicit outcome (`Samples.unresolved`,
`unresolved_probability`; shown in the CLI and GUI); `tail="condition"` samples conditioned on the
resolved sector and records `conditioned_on`; `tail="error"` refuses. `sequence=False` (used by
`run_experiment` and the setup runs) draws only the histogram with one multinomial draw, so up
to 10^12 shots cost no memory per shot; the per-shot sequence is limited to 10^7 shots.

## SEQUOIA setups

`compile_setup_network(setup, library, wavelength, inputs=["pbs.in", ...])` builds the
transfer from an H/V (or `input_basis="DA"|"RL"`) mode pair per input port to an H/V pair per
detector, reusing the components' own Jones transfers and the conventions of `docs/conventions.md`.
Free-space optics, wave plates, phase shifters, polarizers, filters, fibers, fiber splitters and
single-photon detectors reachable from the inputs are converted. A source output port (FockSource,
pair-source signal/idler port) is a valid input; other sources and nonlinear crystals reached
downstream raise `FockCompileError`. Extra arguments: `wavelengths` (one per input, so signal,
idler and Fock photons accumulate `2 pi OPL / lambda` at their own wavelength), `bases` (per-input
Jones basis, first column = the photon polarization), `seed_modes`, `species` labels and
`detector_kind` (`"pnr"`, `"threshold"` or a dict per detector). Spatial-mode overlap between
recombining paths is taken as perfect; fiber coupling uses the source mode. There is no spectral
or temporal structure, so a HOM *delay scan* is not available (use the rate engine); a timing
offset such as the crystal walk-off is not seen.

### The drawn circuit as an experiment (`sequoia.fock.from_setup`)

`build_experiment_from_setup(setup, library, wavelength=None, n_max=None, duration=1.0,
detectors_mode="setup", should_cancel=None, cache=None, classical=None, limits=None,
sources=None) -> SetupExperiment` finds the inputs and detectors itself and
`run_setup_experiment(se, shots, seed, backend)` returns a `SetupResult` (detected-state
probabilities, photons arriving per detector and per mode, expected counts, samples).

- **FockSource**: `photons_per_pulse` N photons, `distinguishability_tag` (equal tags interfere),
  polarization from `polarization_angle` (the photon polarization is the first mode of the input
  basis, so a single-term Fock state), own wavelength; window = `1 / repetition_rate`.
- **Pair sources** (SPDC crystal, entangled source), via the rate engine's emission: the
  single-pair polarization state is the polarization density matrix of the pair amplitude summed
  over members and frequencies, normalised; a mixed state becomes an `InputMixture` of its
  eigenstates. Signal photons enter at `emission.signal_port`, idler photons at `idler_port`
  (both in the same port for a collinear source, in different polarization modes). Pair number:
  `pair_number_mixture` with mean `mu = rate * window` (window `1/repetition_rate` if pulsed, else
  `1/dnu`, as in the Gaussian engine) and `K` = Schmidt number of the pair amplitude with
  polarization traced out (`1 / Tr rho^2`, sparse; `K = 1` with a note if unavailable). A CW source
  has K of the order of the number of lattice frequencies, i.e. nearly Poissonian pair numbers
  and separate modes for separate pairs. `n_max` is chosen as the smallest number of pairs with an
  omitted tail below `1e-4` within `Limits.max_photons`; the omitted weight is reported in the
  result, the inputs table and the Notes.
- **Signal-idler distinguishability**: `eta` = squared overlap of the normalised signal and idler
  spectral intensity amplitudes on the common frequency lattice (phases are not available, so
  this is an upper bound). Each idler photon shares its Schmidt-mode tag with the signal with
  probability `eta`, otherwise it carries a private tag (scalar indistinguishability model).
- Several sources are independent product inputs and must share one window; otherwise
  `IncompatibleWindowsError` lists them (`sources=[ids]` selects a subset). Tags of different pair
  sources never clash with Fock tags.
- **Detectors**: the setup's single-photon detectors reachable from the inputs; SPAD and SNSPD are
  `Threshold(efficiency, dark = 1 - exp(-dark_rate * window))` on the summed H+V photons
  (`detectors_mode="pnr"` gives ideal PNR). Dead time, jitter and afterpulsing are not modelled.
  Counts per second are `probability * windows per second`.
  Each detector's potential click rate times its dead time is checked
  (`SetupResult.dead_time_load`); above 0.1 the result carries a message and the GUI a
  "regime not modelled" line, since the numbers are potential clicks before dead time, not
  recorded counts.
- **Other engines**: a FockSource's light is marked `statistics="fock"`. The rate engine counts
  only its mean flux (`det_data[...]["fock_hz"]`, separate from classical `leakage_hz`) and says
  so (`fock_source_mean_flux`) instead of reporting pump leakage. The Gaussian and Monte Carlo
  engines refuse such setups (`UnsupportedSourceError`) unless `fock_as_poisson=True`, which
  records the Poisson approximation as a `fock_as_poisson` warning.

Approximations: one wavelength per photon species (path phase `2 pi OPL / lambda`); perfect
spatial overlap at recombination; no timing, jitter or dead time; equal-weight Schmidt modes;
pairs in one Schmidt mode are fully bunched, different modes are mutually distinguishable;
the same thermal pair-number law is used for collinear (same-port) sources; a pump power
change only rescales `mu`.

### Run records

`run_record(result)` is a machine-readable record of a setup run (engine and versions, detector
models and dead times, window, windows per second, duration, acquisition model, inputs,
normalization and omitted weight, probabilities, expected counts and sigmas, samples, notes;
non-finite numbers as null). `write_run_bundle(result, folder)` (GUI: Export run...) writes it
as `run.json` with `setup.yaml` as run, the library parts it uses under `library/`, the
photon-number `experiment.yaml` and `summary.txt`, so the run can be repeated without the
author's library.

### Sweeps

`run_sweep(..., fock={...})` (GUI: Sweeps tab, "photon-number observables") evaluates
`fock_metrics` at every point with the given detector mode, pair truncation and duration (the
GUI takes them from the Photon counts tab): `fock:P(d1=n1,d2=n2)` per detector state,
`fock:P_click(d)`, `fock:resolved_mass`, `fock:omitted_weight`, all per detection window. A state
missing at a point has probability 0 there. Link-length and delay axes are refused: the model has
no temporal structure, so such a scan would only change a path phase.

### Listing and extracting the circuit

`list_components(setup, library)` returns one `ComponentRow` per instance (id, type, category,
part, settings and key parameters, ports) and per link. `extract_circuit(setup, library,
wavelength)` / `extraction_experiment(se)` decompose the network on a register with one mode per
(link, polarization) path (a component reuses its input modes): beam splitters become
`BeamSplitter` elements (per polarization), phase shifters and scalar transmissions become
`PhaseShifter` + `Loss`, free-space and fiber links `PhaseShifter(2 pi OPL / lambda)` + `Loss`,
other components one `Block` on their ports' H/V modes, a FockSource at a polarization angle a
rotation block. Elements carry a `label` (component or link id; serialised, shown in the editor).
The transfer matrix is checked against the exact compilation (`Extraction.verified`, deviation
`max_error`); on a mismatch one block holding the compiled transfer replaces it (`fallback`).
The extracted experiment carries the input mixture and detector groups, so it runs unchanged in the
circuit editor; the editor shows a mixture read-only and can drop it (`Use the occupation table
instead`). Extraction uses a single wavelength for all photons.

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

### Preflight estimate

`sequoia.fock.preflight.estimate_experiment(exp, shots=None, backend=None)` returns an `Estimate`
without computing anything: modes, environment (loss) modes, photon sectors, input
configurations (mixture branches, or brightness/indistinguishability branches; bounded by `3^N`
above 20000), permanent work counted exactly as the native engine counts it, working memory,
an upper bound on output patterns, and shots. `errors` are requests the run would refuse or
that would exhaust memory: photons above `max_photons`, work per configuration above
`max_work`, more than 4 GB of working memory, more than 2e6 output patterns, more than 1e12
shots; each names a remedy (lower max pairs / photons / shots, reduce modes). `warnings` flag
long runs and tables that only list the most probable rows. The Photon counts tab shows the
estimate under the run controls (editor: live while editing; setup: after Import or Run) and
refuses a run with errors before starting it ("not run: ...").

Cancellation reaches sampling: `sample(..., should_cancel=...)` checks it before the multinomial
draw and between per-shot chunks of `SAMPLE_CHUNK` (the draws equal one unchunked call). A run
cancelled after its distribution is computed returns `status="partial"` with no samples and the
message `CANCELLED_SAMPLING` (`FockResult.status`, `SetupResult.status`, `run.json`); the GUI
labels it partial in the status line and the model card.

## Code layout

- `fock/setup_sources.py`: source discovery and the input plan (pair-number truncation,
  windows, `list_inputs`); `fock/extract.py`: decomposition into editable elements;
  `fock/records.py`: `run_record`, `write_run_bundle`; `fock/from_setup.py`:
  `build_experiment_from_setup`, `run_setup_experiment`, `list_components`, and re-exports of the
  above.
- `fock/run.py`: run requests and immutable outcomes shared by the GUI (`SetupRequest` ->
  `SetupOutcome` with the estimate, result, extraction, status and the table data
  `SetupViews`, aggregated off the GUI thread with cancellation checks; `run_editor_experiment`).
- `gui/fock_panel.py`: `FockTab` (run controls, lifecycle, files); `gui/fock_editor.py`: the
  circuit editor; `gui/fock_setup.py`: the setup workspace (inventory); `gui/fock_results.py`:
  result views; `gui/fock_common.py`: `ResultTable` (copy of selected cells with Ctrl/Cmd+C or the
  context menu, header tooltips, empty-state text, the most probable 500 rows plus one row with
  the count and probability of the rest, "show more"/"show all"), whole-number cell editors,
  parsers and plots. Plots draw the most probable 40 states and give the rest's probability in
  the title.
