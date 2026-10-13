# ChatGPT Revision — SEQUOIA review and Perceval integration

Review performed 13–14 October 2026 (Europe/Istanbul). SEQUOIA revision reviewed: `1755058c100977f8b8f249e6b70b0e6f7391a276`. Comparison project: neighboring `percevalGUI-v0 0 11`; computational checks used Perceval 1.3.1.

## 1. Main conclusions

SEQUOIA already has a substantial laboratory simulator: classical Jones/ABCD propagation, spectral pair amplitudes, single-pair detector rates, timing Monte Carlo, Gaussian multipair statistics, sweeps, optimization, library provenance, and diagnostics. Its existing suite completed with **1,262 passed and 2 skipped** in 444.03 seconds. This is a useful baseline, but the exploratory checks below reproduced several defects that the suite does not cover.

The first fixes should be:

1. Correct photon-number probability extraction at appreciable gain; the current FFT folds the omitted tail into low photon numbers.
2. Refuse or correctly reconcile Gaussian sources with different time windows; adding an unrelated source currently changes an existing detector's rate.
3. Preserve pulse metadata in saved time tags; the same clicks currently produce radically different accidental estimates after export/reload.
4. Invalidate analysis results and parameter choices when the project changes.
5. Expose the quantum photons in polarization and spectrum displays, rather than presenting only the classical fields.

Perceval GUI supplies a genuinely different capability: arbitrary mode occupations as Fock inputs, distinguishability tags, output-state probability/count/sample distributions, and threshold/PNR/pseudo-PNR detector models. SEQUOIA's Gaussian PNR API does **not** make these features redundant. I recommend an optional circuit/Fock backend behind SEQUOIA's existing document and job infrastructure, with explicit mode, phase, detector, and statistical conventions. Keep the current laboratory engines for the domains they already describe well.

This document is an audit and proposed revision plan. Application/engine source was not changed as part of the review.

## 2. What was actually exercised

### Coverage and evidence

| Area | Checks performed | Result / qualification |
|---|---|---|
| Native SEQUOIA launch | Started the Qt app with macOS desktop access; inspected start-screen gallery and opened Malus' law | Successful |
| Classical GUI | Polarization ellipse/Poincaré display, sweep controls, default failing sweep, corrected degree sweep, optimizer | Successful workflows plus reproducible sweep defects |
| Project switching | Opened Bell-test YAML after a completed Malus sweep | Old sweep target/result persisted; a subsequent sweep had 21 failed points |
| Bell GUI | Rate outcome, Gaussian engine with pump sweep, seeded timing Monte Carlo, polarization/spectrum tabs | Gaussian results populated; Monte Carlo produced 8,366 events in 50 ms, seed 42; quantum polarization/spectrum selectors were empty |
| Library GUI | Opened Library Manager and inspected available controls | Browsing only; native automation intermittently timed out, so destructive library changes were not attempted |
| Shipped engine examples | Classical/rate simulations for Bell, HOM, heralded HBT, MZI and laser/fiber coupling | Results recorded in `engine-probes.json` |
| Full existing suite | `venv/bin/python -m pytest -q --basetemp=/private/tmp/sequoia-chatgpt-retest-20261013` | 1,262 passed, 2 skipped; log retained |
| Additional physics probes | Exactly solvable thermal PNR, independent sources with different windows, source without tagged detectors, gain/background behavior, zero-light CAR | Multiple reproduced defects |
| Recorded-tag analysis | Pulsed export/reload, delayed herald detection, empty records | Reproduced metadata, peak-search and zero-light inconsistencies |
| Perceval computation | Built all seven supplied JSON circuits; SVG rendering; indistinguishable/distinguishable two-photon inputs; threshold, PNR, four-wire PPNR and layered PPNR; a four-photon five-mode distribution | Successful |
| Perceval native GUI | Started `main.py`; attempted native accessibility attachment | Process launch succeeded, but attachment repeatedly timed out. No claim of a completed live Perceval mouse/keyboard walkthrough |

Evidence is in [`docs/review-evidence`](docs/review-evidence):

- `pytest-full.log`: captured full-suite result.
- `engine_probes.py`, `engine-probes.json`, `engine-probes.stderr`: reproducible engine observations.
- `tag_probes.py`, `tag-probes.json`: timing/statistics reproductions.
- `perceval_probes.py`, `perceval-probes.json`, `perceval-probes.stderr`: Perceval computational checks.

Run SEQUOIA probes from this folder with `venv/bin/python docs/review-evidence/engine_probes.py` and `venv/bin/python docs/review-evidence/tag_probes.py`. Perceval probes require its declared `requirements.txt` in a separate environment and are run with that environment's Python. They are investigative scripts, not a claim that regression tests for the proposed fixes already exist. Some output contains Python JSON extensions `NaN`/`Infinity`, intentionally preserving the observed non-finite results.

### Environment findings, not product defects

The initial sandboxed desktop launch failed to connect to macOS pasteboard/window services. Relaunching with desktop access worked. The first suite run had temporary-directory and image-write failures while the disk was almost full; a later run completed cleanly. Do not classify those original failures as optical-engine bugs. Perceval sandbox computation warned about unwritable user cache directories and used temporary caches; computation still completed. An earlier desktop launch was blocked when automatic approval review could not run because of an account usage limit; that was a review-service failure, not a finding that the application was unsafe.

Native accessibility was intermittent. Canvas dragging, trackpad gestures, all keyboard combinations, real-display library import/export editing, and every save-error path were not exhaustively tested live. Existing GUI tests supply broader automated coverage, but they do not replace those remaining manual checks. Findings below identify their evidence level.

## 3. Bugs and inconsistencies

Priority: **P1** means incorrect scientific results or misleading results from a different experiment; **P2** means a broken workflow, hidden limitation, or significant inconsistency; **P3** means presentation/documentation polish. “Reproduced” means a live or numerical check observed the behavior. “Source-confirmed” means the data flow was inspected but the complete user workflow was not reproduced.

### R01 — P1: PNR truncation aliases the photon-number tail into low counts

**Evidence: reproduced against an analytic distribution.** Location: `src/sequoia/engines/gaussian.py`, `_pnr` (around line 503).

For one thermal marginal with mean photon number 10 and cutoff 12:

| Quantity | Current result | Correct analytic value |
|---|---:|---:|
| Vacuum probability P(0) | 0.1279804762 | 0.0909090909 |
| Probability mass retained in bins 0…12 | 1.0000000000 | 0.7103356203 |
| Tail above 12 | Not reported | 0.2896643797 |
| Mean inferred from returned bins | 4.6987919103 | Full mean is 10 |

The mean of a genuinely truncated distribution is also below the full mean; that fact alone is not the defect. The wrong vacuum probability and falsely complete probability mass prove aliasing. Sampling a probability-generating function at only `cutoff + 1` roots of unity returns sums `P(n) + P(n + M) + P(n + 2M) + …`, where `M = cutoff + 1`. It does not extract an uncontaminated truncation. The current low-gain test has a negligible tail and misses this.

**Fix:** evaluate an adaptively enlarged contour/FFT grid with a demonstrated alias bound, or use stable coefficient extraction. Return retained mass, tail/error bound, cutoff, and convergence information. Never normalize a truncated distribution without explicitly calling it conditional. Guard `(cutoff + 1)^number_of_channels` allocations and poll cancellation in PNR evaluation; currently `_pnr` has neither a dedicated allocation guard nor cancellation polling.

**Acceptance:** verify geometric thermal distributions at means 0.01, 1, 10 and larger; repeat with several cutoffs. Low bins must converge independently of cutoff, and retained mass plus tail must agree with one. Add two-channel squeezed-state and loss tests. Normalization alone is insufficient.

### R02 — P1: independent sources with different Gaussian windows change existing rates

**Evidence: reproduced.** Location: `engines/gaussian.py`, `prepare_gaussian`, `GaussianSource`, `GaussianModel` (around lines 321 and 441).

A CW source with `dnu = 1 GHz` reports its existing detector mean rate as **100,000.004846 Hz**. Adding a completely separate source/detector with `dnu = 2 GHz` changes that original rate to **200,000.009692 Hz**. No optical connection or original source setting changed.

`prepare_gaussian` selects the shortest source window as the result window, but each source retains the photon statistics of its original window. Dividing those original means by the new shorter window doubles the first rate. The `gaussian_mixed_windows` warning identifies the situation but does not make the numbers physically consistent.

**Fix:** initially reject incompatible windows with a clear error rather than emit incorrect “exact” rates. Subsequently define an explicit common observation window and a temporal source model: pulse clock synchronization, independent CW coherence intervals and background integration must be treated consistently. Multiplying/rescaling generating functions may be justified for independent intervals; simply changing the denominator is not.

**Acceptance:** adding a disconnected source must not alter an existing channel's singles, marginal PNR, or same-source statistics when expressed for the same observation window. Test unequal CW lattice periods, unequal pulse repetition rates, and CW/pulsed combinations separately.

### R03 — P1: pulsed time-tag export/reload changes accidental estimates

**Evidence: reproduced with seeded, independent pulsed clicks.** Locations: `analysis/correlation.py`, `write_tags`, `read_tags`, `analyse_tags`; `gui/montecarlo_panel.py`, `save_tags`; `cli.py`, `_analyze_tags`.

At 10 MHz, a 1 ns coincidence window and the same saved click arrays:

| Quantity | Original analysis with pulse metadata | Standard reload without pulse metadata |
|---|---:|---:|
| Coincidence counts | 962 | 962 |
| Accidentals | 100,900 Hz | 984.163 Hz |
| Estimated true rate | −4,700 Hz | 95,215.837 Hz |
| CAR | −0.0465808 | 96.7480 |

The generated channels are independent Bernoulli detections on a common pulse clock. The small negative original excess is an ordinary finite-sample fluctuation around zero, not a reason to clamp or reinterpret the record. The reload falsely suggests a strong correlation.

`write_tags` stores only duration and channel arrays. The GUI discards `TimeTags.repetition_rates` during save. `read_tags` cannot restore them, and the CLI calls `analyse_tags` without repetition rates even when a setup is supplied. Analysis therefore switches from pulse sidebands to the CW singles-product estimator.

**Fix:** version the tag format and save repetition frequencies, pulse/reference-clock metadata, timing origin, seed, setup revision and analysis conventions. Keep metadata separate from channel arrays. Offer an explicit pulsed/CW choice for legacy/CSV records; infer a clock from the setup only when unambiguous and record that inference. Do not silently default a known pulsed experiment to CW. For several sources, a single global `max(repetition_rates)` is also insufficient; assign clocks to relevant channel/source contributions.

**Acceptance:** export/reload must preserve counts, centers, accidental model and CAR. Include pulsed/CW and multiple-clock cases. Check old-format compatibility and ambiguous metadata errors.

### R04 — P1: analysis results and sweep targets remain attached to an old project

**Evidence: sweep reproduced live; Gaussian/Monte Carlo retention source-confirmed.** Locations: `gui/sweep_panel.py`, `SweepTab.set_outcome`; `gui/gaussian_panel.py`, `GaussianTab.set_outcome`; `gui/montecarlo_panel.py`, `MonteCarloTab.set_outcome`; `gui/results.py`, `clear`.

Reproduction: open Malus' law, run a valid degree sweep, then open `setups/bell_test.yaml` in the same main window. The sweep still showed `settings:laser.polarization_angle`, output `link:s1.dop`, and the old “21 points” status although those IDs do not exist in Bell. Pressing Sweep produced **21 points, 21 failed**.

`SweepTab.set_outcome` reloads targets only when the candidate list is empty; once a sweep result exists it also stops refreshing output metrics. `ResultsDock.clear` calls each tab's `set_outcome(None)`, but Gaussian and Monte Carlo implement that method as `pass`, so their stored outcomes/tables survive. Long-running jobs also need protection against completion after the document changes.

**Fix:** give each document/edit a revision token and each result its originating token. On project replacement clear result objects, tables, plots, status, metric lists and optimizer state; rebuild candidate lists. Cancel obsolete jobs, and discard late completions from a prior token. On ordinary edits either clear results or visibly mark them stale with the original revision/settings. Export must carry the originating setup, not implicitly suggest the current one.

**Acceptance:** run each analysis, open another setup, and verify no old channel/target/output remains. Repeat by changing parameters during a running job and by applying an optimizer result after a project switch.

### R05 — P2: the initially populated sweep has no physical unit

**Evidence: reproduced live.** Location: `gui/sweep_panel.py`, `AxisRow.set_candidates` (around line 87).

In the newly opened Malus template, the first sweep target was an angle, bounds were 0 to 1, and the unit field was empty. Pressing Sweep gave **21 points, 20 failed**. Entering `deg` made all 21 points run.

`AxisRow` starts without candidates. `set_candidates` blocks signals while inserting the real candidates and never calls `_reset_range`. The range/unit reset normally runs through `currentIndexChanged`, which has been suppressed. Bare zero is accepted as an angle, explaining why exactly one initial point survived.

**Fix:** initialize a valid selection and call `_reset_range` after populating an empty/new target. Preserve user bounds only when the selected target and compatible unit remain the same. Preflight dimensional validity once before scheduling a sweep rather than spend a whole run discovering the same error at every point. Surface per-point failures with their causes.

**Acceptance:** the initial angle target uses degrees; length/delay targets use their correct units; reloading and project switching reset invalid choices. A deliberate mismatched unit is rejected before launching the worker.

### R06 — P2: tag analysis cannot find a herald correlation outside ±50 ns

**Evidence: reproduced.** Location: `analysis/correlation.py`, `DEFAULT_SEARCH`, `find_peak`, `analyse_tags`; analysis inspector center controls.

A deterministic record with 100 heralds and matching detections delayed by 100 ns / 101 ns returns zero herald doubles, zero triples and NaN heralded g2 with a 2 ns window. The underlying matching events are present. The auto peak search is fixed to ±50 ns. `heralded_g2` itself accepts separate centers/search arguments, but `analyse_tags` does not pass user-configurable centers or search ranges to it. The inspector also offers a generic center that is not consumed for this tag-analysis branch.

**Fix:** expose independent `center_a`, `center_b`, search range and delay sign convention; use known setup cable/optical delays to center searches when available. Support automatic searches around nominal delays, not only zero. If the search finds no supported peak, report the search bounds and a diagnostic rather than merely return empty doubles. Preserve these choices in recorded-data analysis.

**Acceptance:** same records with 0, 10, 100 ns and microsecond delays give identical delay-corrected coincidence/g2 estimates when their selected windows contain the peaks. Test asymmetric herald delays and explicit center handling.

### R07 — P2: dark/empty experiments report infinite CAR

**Evidence: reproduced in Gaussian and tag analysis.** Locations: `engines/gaussian.py`, `analyse`; `analysis/correlation.py`, `coincidences`.

Zero true coincidences and zero accidentals yield `car = inf` and `car_total = inf`. This suggests exceptional source quality for a record with no photons. The rate engine already distinguishes this undefined case and the conventions document states CAR should be NaN when both numerator and denominator vanish.

**Fix:** share a statistic helper: no signal/no denominator means undefined; positive signal with exactly zero accidental expectation may mean infinity. Present “undefined: no events” in the GUI. For measured zero sidebands, distinguish an unobserved denominator from proven zero physical background and report an interval/limit when appropriate.

**Acceptance:** empty data, blocked sources, background-only data and positive true signal/zero modeled accidentals behave consistently across engines and exports.

### R08 — P2: Gaussian execution of a source without tagged detectors raises an internal array error

**Evidence: reproduced.** Location: `engines/gaussian.py`, `_GaussianEngine.source` / `prepare_gaussian`.

A setup containing an entangled pair source without tagged detector channels raises `ValueError: need at least one array to stack`. This is a reasonable intermediate state while constructing an experiment. The error comes from stacking an empty detector-transfer list, rather than a meaningful topology/capability diagnostic.

**Fix:** preflight source/channel availability. Either return an explicit empty result with “no tagged photon detectors” or a structured `SetupError` explaining the detector-to-tagger connection required. Preserve existing validation warnings about unconnected source outputs.

**Acceptance:** new blank documents, source-only documents, untagged detector documents and fully connected experiments all terminate cleanly with actionable status.

### R09 — P2: polarization and spectrum tabs do not inspect pair photons

**Evidence: reproduced live on Bell and confirmed in the data flow.** Locations: `gui/analysis_tabs.py`, `LinkTab.link_choices`, `PolarizationTab`, `SpectraTab`; `gui/beamcard.py`, `build_link_infos`.

The Bell experiment has nonzero quantum photon flux, but its Polarization and Spectra link selectors were empty. The engine probe confirmed `classical.links['s_a1'].end is None` while that link carries pair photons. These tabs require a classical beam/mode; the rate layer supplies a flux, but the displayed Stokes vector/spectrum is still taken from the classical light. In pumped SPDC networks that can mean inspecting residual pump light instead of the signal/idler photons.

**Fix:** add an explicit displayed population: classical field, signal marginal, idler marginal, total photon marginal, or herald-conditioned photon. Supply wavelength-resolved reduced coherency matrices and photon flux spectra from the quantum transfer engine. Label quantities with photon-rate/density units rather than classical watts when appropriate. Never infer one source's wavelength for every link in a multiple-source setup.

For Bell photons before an analyzer, each individual reduced polarization state is unpolarized (DOP zero) despite strong joint polarization correlations. Show that correctly; a single ellipse cannot represent entanglement. Provide pair density matrices/correlations separately.

**Acceptance:** Bell source arms show nonzero photon spectra and mixed single-photon polarization; an ideal polarizer conditions the transmitted marginal toward a pure state; pump and pair spectra remain separately selectable.

### R10 — P2: Gaussian-specific diagnostics are hidden from the GUI's diagnostic table

**Evidence: source-confirmed.** Locations: `gui/gaussian_panel.py`, `_done`; `gui/montecarlo_panel.py`, `_done`; `gui/results.py`, diagnostic filling.

Gaussian results contain their own `issues`, including unsupported exchange sources, truncation and mixed windows. `_done` fills its table but does not forward these issues to the main diagnostics. For truncation the status even says “see diagnostics,” although that engine's diagnostic list is not merged into the main table. Monte Carlo `TimeTags.issues` also does not get a dedicated diagnostic presentation.

**Fix:** show diagnostics per engine/run with their originating revision, skipped sources and approximation notes. Include errors/warnings in exported results. Critical unsupported behavior should block execution or mark the result explicitly incomplete. The Gaussian “exact” help text needs a qualifier linking to the supported state, detector, window and timing assumptions.

**Acceptance:** intentionally trigger Gaussian truncation, incompatible windows and an unsupported exchange source. Each message is visible, identifies the source, and can select its component; a skipped source cannot look like a valid dark experiment.

### R11 — P2: the advertised Gaussian pump multiplier leaves classical leakage fixed

**Evidence: numerical behavior reproduced; physical implication depends on which source the control means to scale.** Locations: `engines/gaussian.py`, `GaussianModel`, `prepare_gaussian`; `gui/gaussian_panel.py` pump control.

Background is computed once from detector dark counts plus classical leakage. `gain_scale` then changes squeezing but leaves this background untouched. A coherent-light-only probe returned the same 3,926.6109 mean background photons per chosen window for scales 0, 1 and 10. The no-pair-source fallback window is 1 second, so these are not typical laboratory gated click probabilities either.

Holding dark counts fixed is correct. Holding leakage from the scaled physical pump fixed can give misleading CAR curves. The current control is labeled “pump x” and the CLI calls it a pump-power factor, so its semantics are broader than its implementation.

**Fix:** either rename this to a pair-source gain multiplier and visibly state that classical fields are fixed, or implement source-specific pump-power scaling that recalculates relevant classical leakage/path effects while preserving detector dark rates. Require an explicit observation window for backgrounds without a pair-source clock. Clarify whether reported `pairs_per_window` is a low-gain parameter or the actual high-gain mean; `sinh²(r)` is nonlinear.

**Acceptance:** independently vary dark background and pump leakage; physical pump-off leaves dark counts but removes its optical leakage. A gain-only mode must report its fixed-background assumption.

### R12 — P2: Monte Carlo's pulse statistics and comparison assumptions are not exposed in the GUI

**Evidence: source-confirmed documented limitation, not an unexplained stochastic bug.** Locations: `engines/montecarlo.py`, `simulate_timetags`, `_emission_times`; `gui/montecarlo_panel.py`; `cli.py`.

Monte Carlo defaults to Poisson pair numbers. The API accepts a single `schmidt_number` for negative-binomial pulsed pairs, but the GUI/CLI run paths do not expose/pass it. CW thermal bunching is omitted. Thus a multimode pulsed SPDC rate/Gaussian comparison is not automatically testing the same source statistics, even though the UI presents the analytic comparison as a general check.

**Fix:** expose the emission model and its applicability. Derive per-source Schmidt weights when the model justifies it. A single effective K gives a negative-binomial model only under suitable mode assumptions; for unequal occupations, use the product of mode generating functions. Preserve shot-to-shot ensembles versus simultaneous incoherent modes. At high gain, independent samples of single-pair outcomes miss collective multipair interference: either draw compatible joint Gaussian/Fock outcomes per pulse before timing effects, or reject/flag that regime.

**Acceptance:** pulsed thermal/geometric, Poisson and unequal-Schmidt-mode sources agree with their analytic factorial moments. Comparison rows identify statistic/model differences; systematic model disagreement is not treated as an ordinary sampling z-score.

### R13 — P2: analysis export write errors escape GUI handlers

**Evidence: source-confirmed, not a live crash reproduction.** Locations: `gui/montecarlo_panel.py`, `save_tags`; `gui/analysis_tabs.py`, `save_text`; sweep CSV/PNG export paths should be audited similarly.

After a path is selected, `save_tags` calls `write_tags` directly; `save_text` uses an unguarded `open`. Permission errors or a full disk can escape the Qt callback rather than give a useful “Cannot save” message. This matters given the disk-pressure incident encountered during the review. The main document save already has more deliberate error handling; reuse that approach.

**Fix:** catch filesystem/format errors at GUI boundaries, preserve the result, show the filename and underlying reason, and use atomic replacement for exported artifacts where applicable. Validate suffixes before writing. Do not report success until the write finishes.

**Acceptance:** force a write into a nonexistent/unwritable parent and simulate disk/write failure in a test; the application stays usable and the previous artifact survives a failed replacement.

### R14 — P3: polarization diagram contrast and label spacing need refinement

**Evidence: visible on the live dark-mode Malus screen.** Location: `gui/analysis_tabs.py`, `_Ellipse.paintEvent`, `_Poincare.paintEvent`.

The ellipse H/V axes and sphere rings were very low contrast against the dark panel. Sphere labels crowded the selected link label; the S3 label was partly clipped at the top. `_Poincare` places labels at projected coordinates 1.2 radii from the center but reserves only a small fixed margin.

**Fix:** reserve font-aware label padding, choose a visible palette role for axes/rings, offset labels away from selected points and verify both dark/light themes at several dock sizes and display scales. Treat numerical Stokes/DOP values as the primary readout when diagram space is small. Tiny floating circularity should display as effectively linear rather than a confident handedness label.

**Acceptance:** 1×/2× display scale, small/large docks and both themes show legible axes and unclipped labels. This is presentation polish; it does not establish a polarization-propagation error.

## 4. Physics revisions and boundaries

### Define the measured quantity before comparing engines

Use a common result schema that distinguishes:

- photon flux before detection;
- mean detected photon number;
- threshold-click probability/rate;
- number-resolved outcomes and detector-response outcomes;
- same-window coincidence probability;
- all-time pair coincidence rate;
- timing-window true excess, accidentals and total counts;
- unconditional versus herald-conditioned quantities.

For a coherent state, a threshold detector has `P(click) = 1 − exp(−ημ)` while a number-resolving detector has a Poisson distribution after loss. For a thermal marginal, `P(0) = 1/(1 + ημ)` and the distribution is geometric. A two-mode squeezed source has correlated pair occupations. These distinctions are directly relevant to Sequoia's current photon-rate versus Monte Carlo click-rate differences.

The Bell baseline showed rate-engine CAR approximately 83,887 for `tt` with a 200 ps window, whereas the live Gaussian run showed CAR approximately 1,677 using a 10 ns source window. Those are different accidental integration windows; they are not a demonstrated 50-fold engine error. Report the window beside every CAR/g2 value and require matched definitions in comparison tables. Gaussian `true_hz` is the covariance excess `(Pab − PaPb)/window`, which is not necessarily identical to the rate engine's single-pair contribution at high gain.

### Detector response should be an explicit model

Add a response matrix `D(k|n)` separate from optical efficiency/loss. Supported models should include threshold, ideal PNR and pseudo-PNR/multiplexed threshold detection. For equal ideal multiplexing over N wires, the probability that two photons produce two clicks is `(N − 1)/N`; the Perceval four-wire test reproduced that distinction. Generalize for unequal splitter ratios, efficiencies, dark clicks and finite channel counts.

Avoid counting efficiency twice: decide whether it lives in the optical transfer or detector response and record that choice. Keep dead time, recovery, jitter and afterpulsing as temporal effects after the photon/detector outcome model, with the appropriate coupling between detectors. A GUI showing an ideal optical PNR distribution is not yet a simulation of a physical SNSPD's resolving capability.

### Polarization: preserve basis conventions rather than replace signs casually

SEQUOIA documents per-beam frames and a reflected `diag(1, −1)` polarization factor. Its beam-splitter convention therefore changes the apparent Bell-state labels relative to a common-basis textbook circuit. Existing tests deliberately cover these choices. I did **not** establish that this convention is a physics bug, and changing one reflection sign would be unsafe.

For integration, attach a defined polarization basis/frame to every port, and map input/output bases when building a common transfer matrix. Verify basis-equivalent probabilities, including HH, VV, orthogonal polarizations, diagonal states and all four Bell states. Geometric rotation/flip in the schematic should be clearly distinguished from the wave plate's physical optical angle. When mapping to Perceval, include Jones blocks in spatial×polarization modes and test any reflection-induced basis transform explicitly.

First-order Stokes measurements do not reconstruct joint entanglement. A useful quantum display should provide single-photon reduced density matrices, herald-conditioned polarization, joint two-photon polarization density matrix, purity, and measurement correlations. Add CHSH/tomography workflows that actually acquire the required settings and show assumptions about losses, background subtraction and fair sampling.

### Spectral, temporal and spatial modes need a shared contract

Retain frequency-dependent Jones transfer and distinguishability from the existing JSA model. A scalar “indistinguishability” parameter is useful for a simplified circuit source, but is not a universal replacement for polarization, spectral, temporal and spatial overlap. Likewise, Schmidt number should be defined for the selected/filtered marginal rather than used as an unspecified global source knob.

Before enlarging engine scope, expose existing limitations in the run result: collection modes are fixed rather than back-propagated, central-wavelength spatial propagation neglects chromatic focal shifts, GDD/PMD and cavity feedback are absent, and absolute uncalibrated SPDC brightness has substantial uncertainty. These are documented approximations, not newly demonstrated implementation bugs. A practical next step is back-propagating collection modes to the crystal and recalculating collected brightness consistently, followed by optional dispersion/PMD for experiments where timing overlap demands it.

## 5. What Perceval GUI does, and what SEQUOIA already covers

Perceval GUI builds an ordered mode circuit from input loss probabilities, component/path loss, phases, crossings, beam splitters and MZI macros. REFRESH constructs an `Experiment`; optional loss-channel merging reduces deterministic path segments before simulation. Its sampler accepts `FockState` or tagged `NoisyFockState`, selects a backend, and returns probabilities, sample counts or ordered sample states. Its “herald” FILTER keeps matching rows **without renormalizing** and retains herald modes in output states; it is a display filter rather than a complete conditional-state operation.

Perceval library capabilities are broader than this particular GUI. The local GUI has no wave-plate/PBS polarization controls and no laboratory beam propagation, SPDC phase matching or physical time-tagging workflow. Compare the actual local frontend, not every capability in the Perceval library.

| Functionality | SEQUOIA status | Perceval GUI status | Revision decision |
|---|---|---|---|
| Free-space beams, waists, apertures, fiber overlap | Implemented | Not represented by this frontend | Retain SEQUOIA engines |
| Classical polarization optics | Jones propagation and plots | Not in local component picker | Retain; expose quantum marginals |
| MZI/beam-splitter interference | Implemented with path phase/coherence | Circuit gates and MZI macros | Add explicit phase/circuit components and macro reuse where useful |
| Arbitrary user Fock occupation per mode | No corresponding source/input workflow found | Implemented | Add optional Fock input type/backend |
| Photon distinguishability labels | Detailed pair amplitude supports overlaps; no arbitrary tagged input UI | `NoisyFockState` tags | Add simplified tagged-source workflow with declared semantics |
| Multipair Gaussian source statistics | Implemented with identified limits | Source-noise circuit model differs | Keep Gaussian engine; do not substitute arbitrary Fock sampling silently |
| PNR output probability arrays | Available through Gaussian Python API, currently aliased at significant tail | PNR and pseudo-PNR detectors plus state tables | Fix Gaussian extraction; add response models and GUI distributions |
| Click timestamps/dead time/jitter/afterpulses | Implemented timing Monte Carlo | No such temporal simulation | Retain and repair metadata/model choices |
| Heralded g2/CAR | Implemented | Output-state filtering, not the same statistic | Keep lab analysis; add conditional-state filtering explicitly |
| Parameter optimization | SciPy objective optimizer | “Optimize circuit” merges losses; it does not tune parameters | Keep existing optimizer; add a separate compiler simplification pass |
| Circuit JSON and output-state JSON | YAML setups; CSV/PNG sweeps; NPZ tags | Implemented | Add a versioned importer/exporter/bridge |
| CHSH and tomography | Python metrics exist; dedicated GUI acquisition/analysis workflow not found | Not implemented in local GUI | Add SEQUOIA workflows rather than expect them from Perceval GUI |

### Verified Perceval results

For a lossless 50:50 Rx splitter:

| Input/model | Outputs |
|---|---|
| Indistinguishable `|1,1>` | `|2,0>` and `|0,2>`, each probability 0.5 |
| Different tags `{0}`, `{1}` | `|1,1>` probability 0.5; each bunched output probability 0.25 |
| Threshold detector on mode 0 | Its two photons are registered as one click |
| Ideal PNR detector on mode 0 | Resolves the two-photon outcome |
| Four-wire pseudo-PNR on mode 0 | Two mode-0 photons give two clicks with probability 0.75, one click with probability 0.25 |

Filtering the indistinguishable distribution for mode 0 = 2 leaves probability **0.5**, not 1. That is correct for its stated unnormalized filter, but a new SEQUOIA conditional-state view must separately show success probability and normalized conditional distribution.

All seven supplied circuit JSON files built and rendered to SVG. `4p5m.json` with input `|1,1,1,1,0>` produced 50 output states with probability sum approximately one. The loss compiler reduced `carolan_general15mzi.json` from 228 to 60 loss channels; zero-loss examples eliminated their explicit zero-loss channels. Those results demonstrate useful circuit tooling, not a validated mapping of SEQUOIA's complete laboratory network.

## 6. Recommended integration approach

Three approaches are reasonable:

| Approach | Benefit | Cost / limitation |
|---|---|---|
| **Optional circuit backend inside SEQUOIA — recommended** | One document/editor, reusable lab characterization, explicit Fock workflow | Requires a mode/transfer compiler and capability checks |
| Export/import bridge to the separate GUI | Fastest first integration; enables existing circuit examples | Two state models/frontends; phase and loss mapping still require verification |
| Rewrite every engine around Perceval | Superficially one framework | Does not replace ABCD beams, SPDC collection/JSA, timing physics; largest risk and unnecessary work |

The design below is a proposal for review, not an already implemented or approved change.

### Stage A — shared mode and transfer representation

Compile a supported feed-forward subnetwork to mode labels and wavelength-resolved complex amplitude transfer blocks. Preserve spatial ports, polarization basis, phase convention, source identity and loss. A passive transfer must be contractive; add explicit loss/environment modes or a supported loss channel when a unitary dilation is needed. Keep the original graph for diagnostics and lab geometry.

Add explicit PhaseShifter and waveguide-loss components, crossings/permutations and reusable MZI subcircuits. A generic path length changes phase and group delay together; an ideal circuit phase shifter needs independently declared behavior. Map Perceval Rx/Ry/H conventions through tested complex amplitudes, not just a shared power reflectivity.

### Stage B — inputs and detectors

Add a separate experiment input schema for arbitrary Fock occupations, distinguishability groups and supported mixed states. Keep physical source brightness in pairs/s/mW distinct from Perceval brightness as an emission probability. They share a name but not dimensions or meaning; importing them directly would be a serious conversion error.

Add detector response types and retain optical efficiency exactly once. Compile unsupported sources, overlapping temporal windows, nonlinear elements and unknown detector responses into explicit capability errors. SPDC should not be converted into `|1,1>` automatically; that would discard vacuum, spectral entanglement and multipairs. A heralded SPDC-to-circuit handoff needs a declared conditional density/state model and its success probability.

### Stage C — results and jobs

Add a Photon States tab with input state, selected backend, probability/count/sample table, conditional selection, success probability, normalization and omitted tail. Store output-state probabilities separately from physical event rates. Show backend approximations and resource estimates. Export backend version, seed where applicable, setup revision, mode map and all conditioning conventions.

Use SEQUOIA's worker/job cancellation infrastructure. Perceval GUI's `run_sampler` currently blocks the GUI thread; do not carry that pattern into SEQUOIA. Compile loss simplifications separately from physical parameter optimization: coalesce deterministic loss along phases/permutations, but do not commute unequal arm loss through mixing beam splitters.

### Stage D — importer and validation

Import the local GUI's versioned JSON into explicit circuit components. Parse fields by type, convert cm and radians, preserve insertion order and convention, and flag unsupported data. Its exported circuit does not currently include sampler inputs/herald settings; import those only from appropriate result/experiment records or request an input choice in the new workflow.

Cross-check before exposing the backend as reliable:

1. One/two photons through lossless and lossy identity, phase, splitter, crossing and MZI networks.
2. HOM suppression for indistinguishable photons and recovery for orthogonal tags/polarizations.
3. Loss before/after mixing, compiler simplification equivalence and photon-number conservation in unitary circuits.
4. Threshold/PNR/pseudo-PNR response and avoidance of duplicate efficiency factors.
5. Matching low-gain pair statistics where SEQUOIA and a declared Fock approximation overlap.
6. Identical predictions after polarization-basis conversion, including Bell states and reflected ports.
7. Resource guards, cancellation, old-document result rejection, reproducible sample seeds and exports.

## 7. Implementation order and remaining checks

| Order | Deliverable | Why |
|---|---|---|
| 1 | R01, R02, R03 scientific-result fixes | Prevent incorrect distributions/rates and misleading recorded-data analysis |
| 2 | R04, R05, R07, R08 state/validation fixes | Stop old-experiment results and easy user-workflow failures |
| 3 | R06, R10, R11, R12 statistic/model transparency | Make timing/gain comparisons physically interpretable |
| 4 | Quantum marginal displays, CHSH/tomography acquisition, R13/R14 polish | Make existing physics discoverable and workflows reliable |
| 5 | Minimal optional Fock backend and validated transfer compiler | Adds the substantive Perceval functionality |
| 6 | Pseudo-PNR, importer/macros and richer source handoffs | Extends the backend after conventions and core checks pass |

Remaining manual checks should include native drag-to-connect, fiber conversion, undo/redo after compound edits, copy/paste ID/probe remapping, keyboard shortcuts on macOS, save/reopen after a library-root change, large setups, cancellation during dense Gaussian/PNR work, simultaneous jobs, disk-error exports and small-window layouts. These are remaining coverage items, not bugs claimed from speculation.

Documentation also needs reconciliation: README still refers to a “planned” Gaussian engine in its Monte Carlo section and conventions retain M3-era statements about M8 limitations despite the implemented M8 engine. Specify limitations by engine/version and use the same definitions in CLI, GUI and exported artifacts.

## 8. External references used

Local source and numerical evidence are the primary basis for this audit. Official Perceval references were used to check terminology and convention boundaries:

- [Perceval NoiseModel](https://perceval.quandela.net/docs/v1.3/reference/utils/noise_model.html): source emission/noise parameter semantics; notably brightness is not SEQUOIA's dimensional pair brightness.
- [Perceval unitary components and beam-splitter conventions](https://perceval.quandela.net/docs/v1.0/reference/components/unitary_components.html): Rx/Ry/H phase conventions and polarization component definitions. This is a versioned reference; actual checks here ran 1.3.1.
- [Perceval detector/noise example](https://perceval.quandela.net/docs/v1.3/notebooks/Error_Mitigation_Tutorial.html): physical pseudo-PNR response and noisy photonic experiments.
- [Perceval legacy changes](https://perceval.quandela.net/docs/v1.0/legacy.html): distinguishability tags and replacement of older threshold-output/source APIs.

The proposed integration deliberately starts with local APIs and a tested adapter. It does not require cloud execution or remote quantum hardware.
