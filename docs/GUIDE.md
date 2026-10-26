# Reading the results panels

SEQUOIA answers two questions about a setup: what does the light look like at every point
(beam, power, polarization), and what will the detectors count. The bottom panel is organised by
research task; every task uses the same tools (tabs), and *All tools (advanced)* shows them all.

## Tasks

Pick a task in the list on the left of the results panel, from View > Task, or from **What do you
want to do?** on the start screen (which also opens a matching template). The task is remembered
between sessions. Each task shows a recipe (what to do, the default engine and its assumptions)
above its tools. The tools are the same widgets in every task, so settings and results carry over.

| Task | Tools | Default engine |
|---|---|---|
| Build check | Diagnostics, Links, Readouts & probes | classical propagation and rate engine, after every edit |
| Beam & alignment | Caustic, Budget, Spectra, Polarization | classical propagation |
| Source characterisation | Rate counts, Histograms, Gaussian, Monte Carlo | rate engine; Gaussian and Monte Carlo as checks |
| Two-photon interference | Rate counts, Photon counts, Sweeps | rate engine (delay-dependent); Photon counts without timing |
| Polarization correlations / CHSH | CHSH, Polarization (joint view) | rate engine at the four setting pairs |
| Photon-number / heralding | Photon counts | Fock backend through the setup's detectors |
| Compare & export | Compare, Sweeps | fits of measured data, sweeps of the setup |
| All tools (advanced) | every tab, in the original order | - |

The strip under each tool states what it shows, a one-line model summary (what is included and
what is omitted) and the approximations reported by its last run. Example: Polarization
correlations on the Bell template opens the CHSH tab with the four setting pairs (a b, a b',
a' b, a' b') mapped to the coincidence analyses tt, tr, rt, rr; *Forecast CHSH* fills E and S and
*Export plan...* writes the labelled acquisition plan.

## The experiment used as an example: heralded HBT with type-II PPKTP

A 405 nm laser pumps a PPKTP crystal. Now and then a pump photon splits into two 810 nm photons,
a **signal** (horizontal polarization) and an **idler** (vertical). A polarizing beam splitter
(PBS) separates them. The idler goes to the **herald** detector. When it clicks, a signal photon
is known to be on its way, so the signal arm becomes a heralded single-photon source. The signal
photon meets a 50:50 fiber beam splitter with a detector on each output (A and B), which is the
Hanbury Brown and Twiss (HBT) test: a true single photon cannot make A and B click together.

Numbers from the shipped example (10 mW pump, generic placeholder parts):

| Quantity | Value | Meaning |
|---|---|---|
| Pairs created | 1.27 million per second | what the crystal emits into the collection modes |
| Herald singles | 603 kHz | clicks per second of the herald detector |
| Signal singles (A or B) | 283 kHz each | clicks per second of each signal detector |
| True coincidences herald x A | 126 kHz | herald and A click within 1 ns because they came from the same pair |
| Accidentals | 171 Hz | herald and A click together by chance |
| CAR | about 736 | true coincidences divided by accidentals: how clean the pair signal is |
| Heralding efficiency | about 0.21 (A given herald) | chance that a herald click is followed by a click in A |
| Heralded g2(0) | about 0.003 | close to 0 means single photons; 1 would be classical light |

## The tools

| Tool | What it shows | How to read it |
|---|---|---|
| **Diagnostics** | Problems found in the setup (pump leaking to a detector, a fiber below cutoff, extrapolated data) and the warnings of every engine run. *Engine* says which model produced a row ('setup' for checks of the drawn setup), *Run* which run ('current' for the latest base simulation). Sorted by severity, then engine. | Click a row to select the culprit. Zero rows is good. |
| **Links** | For every link, at its end: power, beam radius, wavefront radius, polarization purity (DOP), fiber coupling, and the pair photons arriving per second (before detection). | Follow a beam down the table to see where light is lost. *n/a* means the light has no Gaussian beam description (hover for the reason). |
| **Readouts & probes** | What detectors and probes read. | Photocurrent for a photodiode, power for a power meter. A probe does not perturb the light; drop it from the parts list on a link (or select a link or part first and double-click the probe). |
| **Rate counts** | Key metrics first: CAR, heralding efficiency, heralded and unheralded g2(0), two-photon interference visibility = 1 - C/C_distinguishable, and the polarization correlation E = (C++ - C+- - C-+ + C--)/sum when four analyses form a 2 x 2 set; each with its definition and denominator. Then per detector: potential clicks before dead time (detected photons plus dark counts), recorded singles after dead time and their pair and background parts, the dead-time survival, and the coincidence analyses. | Model: first order in the pair rate, non-paralyzable dead time (detector then tagger), Gaussian jitter; multi-pair only through the uncorrelated accidental formula (use Gaussian for high gain). Undefined values say why (e.g. "undefined: no accidental estimate"). |
| **Caustic** | Beam radius along the path. | A dip is a focus. Check that the beam is not larger than the next part. |
| **Polarization** | *Single link*: the polarization of the light on a link as an ellipse and a point on the Poincare sphere (drag to rotate), for the classical field or the reduced state of one photon. *Joint (pairs)*: the two-photon density matrix of a pair, its entanglement metrics, and the state of photon b given that photon a passes a chosen analyser. | A single photon of an entangled pair is unpolarized (point at the centre); the correlation is only in the joint view. Linear states sit on the equator; +S3 is L, -S3 is R (see docs/conventions.md). |
| **Spectra** | Power versus wavelength on a link, or the photon-rate spectrum of the signal/idler photons (second selector). | Filters show up as cut or narrowed spectra. |
| **Histograms** | Arrival-time difference between two detectors. | A peak at zero delay is correlated pairs; the flat floor is accidentals. Its height over the floor is the CAR. |
| **Budget** | Power or photon flux along a path, plus what the result would be with each part replaced by an ideal one. | Parts at the top of the list cost the most; improve those first. |
| **Sweeps** | Vary one setting (a wave-plate angle, a lens distance) and plot a result; the optimizer finds the best value. | A sweep of the half-wave plate in the Bell test shows the polarization fringe. |
| **Monte Carlo** | A simulated stream of individual detector clicks with jitter, dead time and noise. | It should match the analytic Rate counts within the shown error (z within 3). Use it when dead time or timing matters. |
| **Gaussian** | Multimode Gaussian photon statistics per window when several pairs can be created at once (no dead time or jitter). | CAR falls and heralded g2 rises as you turn up the pump; this is why the pump power is kept low in these experiments. |
| **Photon counts** | Photon-number statistics of the drawn circuit. *Import from setup* lists every part and link and the photon inputs (Fock sources: N photons, tag; pair sources: mean pairs per window mu from the pump, Schmidt number K). *Run* gives the output-state probabilities through the setup's detectors, potential click rates before dead time and expected counts over a duration (binomial sigma for fixed windows), click patterns, photon-number marginals, heralded (conditional) distributions with their success probability, samples, a plot and the approximations used. *Extract to circuit editor* converts the circuit to editable elements. | Equal photons on one beam splitter bunch (no coincidence); a phase shifter in an interferometer arm moves the two-photon fringe; the omitted-weight note tells you when the truncation matters. Timing, spectral shape and dead time are not modelled (use the rate engine for a HOM delay scan). |
| **CHSH** | Bell-test forecast: analyser parameter and two settings per arm, the four coincidence analyses, counts per setting pair, E and S raw and accidental-subtracted with the counting spread. | A forecast under fair sampling, not a measured violation; the time is per setting pair. |
| **Compare** | Measured data against fits or a sweep of the setup (see below). | Not-identifiable parameters are flagged instead of reported as unique values. |

## Moving around the canvas

- Drag the empty background to move around. Hold Shift and drag to select several parts.
- Scroll or pinch to zoom; the buttons at the bottom-left of the canvas zoom and fit the view.
- Drag from a port to another port to connect two parts. Select a link to switch it between
  air and fiber, or to set the fiber type and its loss.

## Saving and starting another experiment

- **File > Save** (Ctrl+S) writes the experiment to its `.yaml` file; a new or template-based
  experiment asks for a name first, and example files shipped with SEQUOIA are never overwritten.
- **File > New Experiment in New Window** (Ctrl+Shift+N) opens the start screen next to the
  current experiment so you can work on two at once.
- **File > Close Experiment and Go to Start Screen** (Ctrl+Shift+W) asks to save, closes the
  experiment and shows the start screen with templates and recent files.

## Part properties that are modelled

Every optic can carry the quality data of a catalogue "specifications" tab:

| Property | Where | Effect |
|---|---|---|
| `wavefront_error` (RMS waves at 633 nm) | lenses, mirrors, splitters, plates, filters, couplers | raises the beam's M^2 (worse focusing and fiber coupling), keeping the local beam size |
| `damage_threshold_cw` (W/cm^2), `damage_threshold_pulsed` (J/cm^2) | same optics | warning at 50% and when exceeded, computed from the beam size on the part |
| `clear_aperture` | lenses, mirrors, couplers, iris | clipping loss and warning |
| `focal_length_curve` | lenses | chromatic focal shift (singlets use the N-BK7 or fused-silica Sellmeier curve) |
| `order`, `retardance_error` | wave plates | multi-order plates are far more sensitive to wavelength; as-built tolerance such as lambda/300 |
| `beat_length`, `polarization_extinction_db` | PM fibers | the fiber acts as a long retarder along its slow axis, with polarization crosstalk |
| `transmission`, `reflectance`, ... curves | filters, mirrors, splitters | full spectral dependence; `angle_of_incidence` blue-shifts filter edges |

| `FockSource` (`photons_per_pulse`, `distinguishability_tag`) | sources | exactly N photons per pulse: classically a pulsed beam of mean power N h nu f, and the occupation read by the Fock backend; emits no pairs |
| `PhaseShifter` (`phase`, `axis` both/H/V) | polarization | explicit phase e^{i phi} on both polarizations or on one linear component; a phase of pi equals a lambda/2 change of the link length in that arm (path phase is already modelled, do not add both) |

Splitters whose transmitted and reflected fractions add up to more than 1 are rejected.
`scripts/make_catalog.py` regenerates the typical-class parts (filters, lenses, mirrors, wave
plates, polarizers, fibers, detectors, lasers) with these properties.

## Comparing with measured data

The **Compare** tab loads a CSV, TSV or XLSX file (the file is never changed). Map the columns:
x and y with their units (taken from headers such as `delay (ns)` when present), optionally a y
uncertainty, an acquisition-time column or one acquisition time, and a background (sideband)
column. Rows with missing values are dropped and listed. *Subtract background column* makes a
derived curve that keeps the original and records the operation; the background is the mean of
the given number of sideband windows, so its counting variance B/n is added.

Weights: the given uncertainties; otherwise Poisson from counts (or a rate and its acquisition
time) using the model's expected counts, iterated to the maximum-likelihood fit; otherwise an
unweighted fit with a warning (errors then come from the scatter and chi2 is not a test).

| Fit | Parameters | Notes |
|---|---|---|
| Polarization fringe | amplitude, phase, offset; visibility = A/offset | cos 2x for an analyser angle, cos 4x for a half-wave-plate angle |
| Coincidence peak | center, width (Gaussian sigma), background, area (+ exponential tail) | x must be a time |
| HOM dip | delay offset, visibility, width, baseline (Gaussian or sinc) | x is a time or a stage length |
| Singles vs pump | brightness, efficiency, dark | brightness and efficiency are reported as not separable |
| Compare with a SEQUOIA sweep | optional loss-budget scale factor | sweeps the drawn setup at the data x values |

The table lists each value with its standard error (from the covariance at the solution) and a
status: fitted, fixed (type `name=value` in data units to hold an independently calibrated
value), derived or **not identifiable**. A fit is flagged when the Jacobian is rank deficient
(only a combination of parameters is constrained), a standard error exceeds the value, or two
parameters correlate with |r| > 0.95. *Export* writes the data, model and residuals (CSV) and a
JSON record with the file hash, mapping, dropped rows, derived operations and the fit. From
Python: `sequoia.analysis.measured` (`load_measured`, `compare`, `compare_sweep`) and
`sequoia.analysis.fitting` (`fit`, `fringe`, `peak`, `hom_dip`, `scale_model`, `singles`).

## Calibration records

A part folder may hold `calibration.yaml` next to `part.yaml`: measured (or vendor nominal, or
model) values kept apart from the nominal part. Each record names the `parameter`, a `value` with
`std` or a `curve` CSV (a third column is the per-point std; `covariance` names an n x n CSV), and
optionally `date`, `conditions`, `local_id`, `measured_by`, `method`, `validity` and `note`. A
record replaces the nominal value only when it ranks higher (measured > vendor nominal/datasheet >
model > ideal default; a value without provenance counts as nominal), and the newest record of
the highest rank wins. `zero_offset` (wave plates, polarizers) and `timing_offset` (detectors) are
used by the uncertainty propagation only. The Library Manager shows the records read-only with
their status and the part fingerprint (a hash of the part's files that changes with any
calibration).

```yaml
records:
  - {id: od_2026_10, parameter: optical_density, value: 1.05, std: 0.02, date: 2026-10-01,
     method: power ratio at 633 nm, local_id: ND-17, measured_by: BK}
```

`sequoia.library.calibration.placeholder_report(setup, library)` lists the critical parameters
of a setup as measured, vendor nominal, model, ideal default, untraced or setup override.
`propagate_uncertainty(setup, library, "probe:pm.power_W", ["nd.optical_density"], n=200,
seed=1)` draws the declared calibration uncertainties (efficiencies, losses, optical density,
power, wave-plate zero offsets, timing offsets) and returns the spread of the metric and each
parameter's one-at-a-time contribution. Counting noise is not included.
