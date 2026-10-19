# SEQUOIA — second-round revision review

Review completed: 20 October 2026. Code inspected and exercised on 19–20 October 2026.

Reviewed commit: `d8c0f6c500f1cfdc35f04b5f7da55a8698849950`. First-review baseline: `1755058c100977f8b8f249e6b70b0e6f7391a276`.

## Assessment

The revisions make substantial progress. SEQUOIA now has a native Fock circuit backend, an optional Perceval bridge, setup-to-circuit extraction, explicit detector response models, photon-count distributions, pair-number mixtures, and a CHSH calculation. Several original numerical and GUI defects have targeted fixes and regression tests.

However, the revised application is not yet consistently safe to interpret as an experimental prediction. The largest remaining issues concern preserving the meaning of an experiment across editors, distinguishing different statistical models, retaining the correct normalization, and binding results to the exact inputs that produced them. These matter more than adding another engine or optical component.

My recommended next milestone is **a reproducible experiment workflow**: choose a research question, configure the measurement, run a compatible model, inspect its assumptions and uncertainty, compare against data, and export a record that another researcher can reproduce.

This report distinguishes reproduced defects, source-confirmed gaps, documented approximations, and proposed improvements. Application code was not modified.

## Evidence and coverage

Evidence lives in `docs/review-evidence-round2/`:

- `probes.py` and `probes.json`: independent numerical and headless Qt probes of sampling, timing metadata, CHSH, result lifecycle, and editor persistence.
- `detector_probes.py` and `detector-probes.json`: dark-count response and detector/editor round-trip probes.
- `mixed-clock.npz`: synthetic exported clock-metadata reproduction; no laboratory data.
- `pytest-full.log` and `pytest-focused.log`: interrupted test runs. Neither contains a completion summary. They are **not evidence of a passing full suite**.
- `pytest-resume.log`: bounded, sequential verification after the application was force-quit. Its outcome is recorded at the end of this report.

Live desktop coverage: launched `templates/fock_hom.yaml`, inspected diagnostics and the Photon counts workspace, ran the drawn setup, inspected output states and detector clicks, extracted it into the circuit editor, and ran the extracted circuit. Extraction reported 18 elements, four modes, and a maximum transfer-matrix deviation of `5.8e-11`. The drawn setup and extracted editor both produced four detector outcomes, but the editor lost detector dark counts, as explained below. The interface screenshot also showed crowded table headings, a long row of analysis tabs, and low-contrast help text.

An attempt to open the Bell template through the native file picker did not complete reliably; I do not claim a new live Bell/polarization inspection. Those revisions were inspected in source and their tests. The first launch failed because sandboxed macOS pasteboard/window services were unavailable; launching with desktop access worked. That is an execution-environment limitation, not an application defect.

The previous review's 1,262-pass result belongs to the old revision. It must not be used as the current revision's test result. The user's reported memory problem interrupted this review; its cause was not isolated, so it is not attributed to SEQUOIA. Large concurrent tests were not restarted on resumption.

## 1. Verification of the first-round revisions

“Implemented” below means the relevant change is present and matches the intended fix on inspection; it does not imply exhaustive scientific validation. The bounded test result identifies the subset freshly verified after resumption.

| Original item | Current assessment | Remaining work |
|---|---|---|
| R01: Gaussian PNR tail aliasing | Substantially corrected. `_pnr` increases Fourier grid lengths, returns retained mass/tail/convergence metadata, checks cancellation, and limits joint-grid work. Analytic thermal regression cases are present. | Treat the grid-difference estimate as numerical convergence evidence, not a universal rigorous error bound. Add correlated high-gain, cutoff-independence and moment-convergence checks. Do not silently normalize away the tail. |
| R02: incompatible source windows | Implemented: incompatible windows raise `IncompatibleWindowsError`; same-window independent-source tests exist. | Expose the incompatible sources and windows before running, with an explicit source-selection remedy. A common-clock temporal model remains future work. |
| R03: exported tags lose clock | Single-clock round-trip implemented with versioned metadata and atomic writing. | Mixed-clock handling remains incorrect when the per-channel clock map is empty; see RR03. Metadata still needs a complete run/configuration fingerprint. |
| R04: stale project results | Project-reset/generation guards and outdated labels are implemented. | Edit-during-run race and circuit-editor state remain incomplete; see RR02 and RR11. |
| R05: initial sweep unit | Implemented: candidate initialization resets unit/range and preflight validates axis units. | Rebuild candidates when the current document's structure changes, not only when the list is empty. Provide physical bounds and a meaningful default sweep for the selected research task. |
| R06: delayed herald peaks | Implemented: wider automatic search, explicit `center_a`/`center_b`, and peak-found metadata. | Automatic peak selection is a statistical choice. Save search ranges/centers, distinguish searching from measuring, and support fixed centers for confirmatory analysis. |
| R07: zero-count CAR | The affected engine paths use a shared ratio convention; zero/zero becomes undefined. | UI should render “undefined: no accidental estimate” rather than unexplained NaN. Zero measured sidebands do not prove infinite physical CAR. |
| R08: Gaussian with no tagged detectors | Implemented: an empty result and diagnostic replace the array-stacking failure. | An actionable preflight message is better than a completed-looking empty analysis. |
| R09: quantum polarization/spectrum tabs | Implemented in source: population selectors and single-photon marginal displays. Tests exercise Bell-source photon entries. | Preserve the distinction between a single-photon reduced state and two-photon entanglement. Add conditional/joint observables and state provenance. |
| R10: engine diagnostics hidden | Gaussian and Monte Carlo issues are forwarded into Diagnostics. | Bind diagnostics to a run and model, and surface approximations at the affected result. Fock limitations mostly live in Notes. |
| R11: pump multiplier leaves leakage fixed | Implemented: leakage scales with the multiplier while dark counts stay fixed. | A global multiplier is still ambiguous with several pumps or independent background lasers. Add source-specific controls and state exactly what scales. |
| R12: MC statistics transparency | Improved: a pulsed-source Schmidt-number control and model-labelled comparisons were added. | CW/temporal limitations remain. The control permits values between 0 and 1 but `run()` maps those to Poisson; disallow that interval or explain the coercion. |
| R13: export IO failures | Original text/tag export paths now handle errors; tag export is atomic. | Apply the same persistence contract to new circuit files and preserve all their fields. RR01 is a separate scientific-data-loss issue. |
| R14: polarization plot styling | Source now uses font-aware margins and revised rendering. | Not independently closed by a fresh native visual check in this round. Test small windows, light/dark palettes, scaling, and near-linear/circular limits. |

The Perceval-related work is real, not just a UI placeholder. Tests cover matrix conventions, HOM bunching/distinguishability, losses, detector response, and randomized native-versus-Perceval comparisons. The new setup integration also has tests for phase fringes, Bell correlations, Gaussian comparisons, thermal pair-number statistics and extraction. Their presence is valuable, but the interrupted runs prevent claiming that all of them passed in this session.

## 2. Correctness revisions to make next

Priority meanings: **P1** can silently change scientific conclusions or the experiment being evaluated; **P2** produces misleading output or a significant workflow failure; **P3** is a usability refinement. Evidence status is stated for every item.

### RR01 — P1: the circuit editor silently changes imported experiments

**Evidence: independently reproduced; also visible in the live extraction flow.**

`src/sequoia/gui/fock_panel.py:761` (`experiment`) reconstructs detector models from a small set of table fields. `set_experiment` at line 819 does not preserve all the original model fields in an editable backing document.

A pseudo-PNR detector with `dark=0.2`, two wires, weights `(0.9, 0.1)` and wire efficiencies `(0.5, 0.8)` becomes a two-wire detector with `dark=0`, default equal weights and default wire efficiencies after `set_experiment(exp); experiment()`. Named conditions disappear and the experiment description becomes empty. These are not cosmetic changes: the output probabilities and heralding acceptance can change.

In the live Fock template, the drawn setup produced `P(1,1)=0.302501` with dark counts; extraction and execution in the editor produced `0.3025`. That small difference is consistent with the much stronger synthetic round-trip reproduction. The transfer-matrix verification only verifies the optical transfer; it does not verify detector/input/configuration fidelity through the editor.

**Fix:** retain a complete `Experiment` document and apply only explicit edits. Unsupported properties should be preserved with a read-only summary or an advanced editor. Do not reconstruct the document from the visible widgets. Preserve detector ordering/group membership, conditions, description, custom wire parameters, random-seed semantics and count metadata.

**Acceptance:** load/save/reload and setup/extract/editor/run produce identical distributions and conditionals within tolerance, including vacuum with dark counts, asymmetric multiplexers, grouped detectors and multiple named conditions. The test must compare physics as well as serialized keys. Warn explicitly before any intentional lossy conversion.

### RR02 — P1: an analysis completed after an edit can appear current

**Evidence: reproduced through the actual Qt completion handler with controlled revisions.**

`src/sequoia/gui/runguard.py` checks the generation token, which changes on reset, but does not reject a run merely because the document revision advanced. The probe started a Gaussian run with revision 1, supplied revision 2, then delivered the old outcome. `_done` accepted it and displayed `window 1 s, 0 Schmidt modes`, with no outdated marker.

The normal `set_outcome` path can mark an already displayed result outdated. This does not cover an edit while a result is still pending: completion overwrites the status with a fresh-looking message. Tracking only completed base simulations also creates a delay between a document edit and invalidation.

**Fix:** capture immutable document ID, document revision, library/calibration fingerprint and analysis-configuration revision when launching. Compare these on completion. Listen to document edits immediately. Either discard the old result or retain it as a clearly labelled historical run. Changing pump scale, duration, detector mode, seed or circuit-editor input must also invalidate the displayed result's “current” status.

**Acceptance:** start an analysis, edit while it runs, let the new base simulation finish first, then deliver the old result. It must never appear current. Repeat with auto-run disabled, a failed base simulation, library edits, project switches and cancellation. Export must describe the actual run, not the current widgets.

### RR03 — P1: an empty channel-clock map restores the wrong global clock

**Evidence: independently reproduced.**

`TimeTags.clock` in `src/sequoia/engines/montecarlo.py:40` returns `self.channel_clocks or self.repetition_rates`. An empty map can legitimately mean that no channel has one valid clock—for example, every channel has incompatible source contributors. The fallback then reinstates the legacy list.

With an empty map and source rates `[50e6, 80e6]`, analysis used the sideband estimator; export assigned **80 MHz to both channels**. The new metadata format faithfully persists the wrong inferred clock.

**Fix:** distinguish “legacy clock map absent” from “explicit map present but empty”, preferably with `None` versus `{}`. Preserve clock identities and source relationships, not just equal numerical frequencies. Two independent oscillators with equal repetition rates are not necessarily synchronized. Mixed/unknown timing should have an explicit analysis policy and diagnostic rather than a silent global maximum or unjustified CW estimator.

**Acceptance:** cover no valid clocks, one valid channel plus one mixed channel, unequal clocks, equal-frequency unsynchronized sources, CW/pulsed mixtures, and legacy files. Export/reload must preserve both known clocks and the absence of a valid clock.

### RR04 — P1: CHSH reports a violation for an invalid acquisition duration

**Evidence: numerical reproduction on the Bell setup; additional statistical issue confirmed in source.**

`run_chsh(..., integration_time=0)` and `integration_time=-1` both returned `S=2.743574221`, `sigma=NaN`, and `violates=True`. The formatted output says “violates” while also reporting no coincidences for all four setting pairs. `ChshResult.violates` at `src/sequoia/analysis/chsh.py:30` falls back to `S > 2` when `sigma > 0` is false, including NaN.

The method also evaluates only positive S. An equivalent experiment with output labels reversed can give negative S and should be assessed against `|S| <= 2`, with the chosen convention retained in the output.

A separate issue is the default `true_hz` uncertainty. The implementation uses the net/subtracted rate as if its counts were independent Poisson observations. If an experimental estimator is `C_signal - alpha*C_sideband`, its variance includes the sideband contribution; it is not generally equal to the net count. The current result is a model forecast, not an observed statistical rejection of a local model.

**Fix:** validate finite, positive duration and settings; return an explicit undefined significance when uncertainty is unavailable. Separate predicted S, anticipated counting uncertainty, and measured-data significance. Default to raw coincidences for an experimental CHSH workflow; show corrected correlations separately. Use the covariance of the actual estimator, document fair-sampling/postselection assumptions, and use `abs(S)` for the two-sided local bound. State whether the integration time is per setting pair or total; the current computation uses it for each of four pairs.

**Acceptance:** zero/negative/NaN duration rejected; no counts cannot produce a significance claim; output relabelling preserves violation assessment; simulated count ensembles validate error propagation; background subtraction increases uncertainty appropriately. A valid Bell baseline should still reproduce its expected S.

For context, optical Bell-test literature explicitly distinguishes fair-sampling and postselection assumptions: [Tailored two-photon correlation and fair-sampling: a cautionary tale](https://arxiv.org/abs/1308.4819) and [Randomness extraction from Bell violation with continuous parametric down conversion](https://arxiv.org/abs/1805.02828). These are reasons to label the inference precisely, not reasons to remove a useful CHSH forecast.

### RR05 — P2: count uncertainty is inconsistent with fixed pulse trials

**Evidence: independently reproduced.**

`expected_counts` in `src/sequoia/fock/analysis.py` uses `sqrt(N*p)` whereas `outcome_table` uses `sqrt(N*p*(1-p))`. For 100 deterministic pulses and an outcome with probability one, the first returned sigma 10 and the second returned zero. The GUI explicitly labels its column “Sigma (Poisson)”, but that approximation is inappropriate as the default for high-probability outcomes of a fixed-pulse Fock experiment.

**Fix:** make the acquisition model explicit. For a fixed number of independent pulses, mutually exclusive output-state counts are multinomial: `Var(N_i)=N*p_i*(1-p_i)` and `Cov(N_i,N_j)=-N*p_i*p_j`. Poisson is an appropriate selected approximation/model in other acquisition regimes. Uncertainty of an estimated mean, predicted shot-to-shot variability, and uncertainty from calibration are different quantities; label them separately.

**Acceptance:** p=0 and p=1 have zero binomial variance; p=0.5 gives `sqrt(N)/2`; the rare-event limit approaches Poisson; summed/conditional outcomes use the appropriate covariance. Both APIs and the GUI agree when configured for the same acquisition model. [NIST binomial reference](https://itl.nist.gov/div898/handbook/eda/section3/eda366i.htm).

### RR06 — P1: samples silently condition on retained probability mass

**Evidence: independently reproduced.**

`sample` at `src/sequoia/fock/analysis.py:108` samples using `p / total` regardless of omitted input weight. A mixture with retained mass 0.25 and omitted mass 0.75, representing only one known output, generated that output in all 100 samples. Its unconditional probability remained 0.25 and expected count over 100 windows was 25. The Samples and probability/count views therefore refer to different ensembles without an explicit sampling-mode choice.

**Fix:** default to rejecting unconditional sampling when a material tail is unresolved, or include an explicit “unresolved/truncated” outcome. Alternatively expose “sample conditioned on retained sector” and carry that denominator through every label/export. Do not conflate physical loss, no click, and numerical omission. A tail error bound must also accompany heralding success estimates.

**Acceptance:** retained mass 0.25 cannot silently produce 100 unconditional known outcomes from 100 windows. Full-mass distributions retain existing behavior; conditional sampling explicitly reports its conditioning probability. Test both CLI and GUI paths.

### RR07 — P1: PNR dark counts disappear on vacuum because output support depends on input cutoff

**Evidence: independently reproduced.**

For `dark=0.2`, `IdealPNR.response_matrix(0)` and `PseudoPNR(wires=4).response_matrix(0)` both returned `[[1.0]]`; threshold returned the expected `[0.8, 0.2]`. PNR saturates at `n_max`, and pseudo-PNR truncates its output columns before applying the extra dark click. Thus the numerical input cutoff is treated as detector capacity. A dark event cannot create a count above that cutoff, including a single count on vacuum.

**Fix:** separate input occupation cutoff from detector output capacity. Under the current independent-one-dark-click model, ideal PNR needs support through `n_max+1`; pseudo-PNR capacity should derive from its wires and declared dark model. If per-wire dark probabilities are intended, implement their response explicitly rather than adding an arbitrary global count. Update layout/tensor consumers to support rectangular response matrices.

**Acceptance:** vacuum dark counts survive; one photon plus a dark event can produce two PNR counts; increasing an unrelated input cutoff does not change the vacuum response; all response rows sum to one.

### RR08 — P2: a zero-duration Fock experiment becomes a one-second experiment

**Evidence: independently reproduced through `run_experiment`.**

`src/sequoia/fock/experiment.py` selects `exp.duration_s if exp.duration_s else 1.0`. A document/API request with `duration_s=0` produced 100 expected counts at 100 Hz and recorded a duration of one second. The helper itself permits zero, so this is inconsistent semantics, not a uniformly rejected input.

**Fix:** use an explicit `is None` default. Decide whether zero is valid (zero counts) or invalid (clear validation error); apply the same rule in API, document schema and GUI. Reject non-finite durations/rates and validate their units and mutual consistency.

**Acceptance:** `None`, zero, positive, negative, NaN and infinity have documented behavior. No explicit user value is silently replaced by a default.

### RR09 — P2: Fock inputs are presented as classical pump leakage in other engines

**Evidence: observed live and traced to `FockSource.emit` in `src/sequoia/components/sources.py`.**

Opening the supplied Fock template produced seven general diagnostics, including “classical light alone” and “pump_leakage”, plus “no component emitted photon pairs”. `FockSource` emits a mean-power `SourceLight` representation for classical propagation; downstream engines interpret that representation as classical background. A mean-field proxy can be useful for geometry and mean transport, but it does not carry Fock photon-number statistics.

**Fix:** give source populations an explicit statistical kind and backend capability contract. Route Fock number statistics to the compatible engine. Mean-only outputs should be labelled as such and should not classify all Fock flux as pump leakage. Gaussian/MC workflows that substitute Poisson light should refuse the unsupported request or require an explicit, recorded approximation.

**Acceptance:** a pure Fock-source setup does not tell the researcher to block their desired photons as pump leakage. Backend selection explains supported observables. A single-photon input retains antibunching in any workflow claiming to simulate its statistics.

### RR10 — P2: “setup detectors” can display rates above their physical dead-time ceiling

**Evidence: live output; limitation is documented in source/Notes. This is a presentation and model-eligibility defect, not an undiscovered missing physical term.**

The supplied template uses 80 MHz pulses and SPADs with 50 ns dead time. Photon counts, with “setup detectors” selected, displayed about **44.0 million clicks/s per detector** and **24.2 million joint clicks/s**. The general diagnostics separately state a ceiling of 20 million counts/s. The Fock engine intentionally omits dead time, so its independent-window prediction is mathematically consistent with its approximation but unsuitable as an unqualified prediction of recorded detector rates.

**Fix:** label these quantities “potential clicks/s before dead time” or “independent-window prediction”. Put the limitation beside the numbers. Preflight should detect `predicted_rate * dead_time` outside the dilute regime. A physical recorded-rate workflow needs a temporal response model; simply capping the rate or multiplying all coincidences by one scalar is not a generally valid repair. Keep the ideal distribution available for circuit design.

**Acceptance:** this template visibly explains the incompatible acquisition regime before presenting lab-count estimates. Tests distinguish optical arrival, potential detector click and recorded tag rates. Shared dead-time effects and pulsed renewal behavior receive dedicated validation when implemented.

### RR11 — P2: circuit-editor results survive reset and editor changes are not fully tracked

**Evidence: headless Qt reproduction plus source inspection.**

`FockTab.reset` at `src/sequoia/gui/fock_panel.py:563` clears setup results and setup tables, but retains `self.result` and the circuit-editor results table. In the probe, running/resetting/refreshing left the old result accessible and its row visible. Existing project-switch tests cover the setup-mode tables, not this editor path.

A standalone editor may intentionally survive a drawn-project switch. If so, it needs a clear separate document identity, unsaved-change handling and independent result/configuration revisions. At present the mixture of shared and separate lifecycle behavior is ambiguous.

**Fix:** decide ownership explicitly: either make the editor a separate named circuit document or reset all state on project switch. Track every editor change and preserve its run history. Do not display a probability table next to modified circuit fields without a stale marker.

**Acceptance:** project switching, editing, importing, resetting and rerunning have deterministic documented behavior in both modes. Closing or replacing an edited circuit gives recoverable save/discard behavior. A refresh cannot resurrect a reset result.

## 3. What a quantum-optics researcher should use SEQUOIA for

The strongest role is an **experiment-planning and interpretation companion**. A researcher wants to decide what to adjust, what to measure, how long to acquire, and whether the result is consistent with the apparatus. The software should help answer those questions with traceable assumptions.

| Laboratory task | Useful result | Present foundation | Next user-facing workflow |
|---|---|---|---|
| Align a source and couple into fiber | Waist location, coupling budget, tolerances, next useful adjustment | Canvas, caustics, classical propagation, coupling and sweeps | Guided alignment scan with physical control names and comparison to a measured power/coupling curve |
| Commission a heralded source | Singles, heralding efficiency, CAR, heralded g2, multipair contamination | Rate, Gaussian and MC engines | One source-characterization recipe with explicit windows, backgrounds and acquisition model |
| Diagnose a poor HOM dip | Visibility versus delay/polarization, accidentals, imbalance and distinguishability | Rate HOM modeling and Fock interference | A delay-scan workflow that explains which backend has temporal information; attach measured trace and fitted nuisance parameters |
| Characterize polarization entanglement | Correlations, visibility, CHSH forecast, measured S and uncertainty | Marginal polarization views, Bell source, CHSH API/CLI | Four-setting GUI measurement plan and later tomography; clearly distinguish marginal DOP from pair-state quality |
| Plan photon-number experiments | P(n), detector click patterns, heralding success, expected useful events | New Fock/Perceval and detector models | Preserved experiment documents, conditional/unconditional views and truncation-aware sampling |
| Decide whether to change a component | Gain in useful counts versus purity/visibility, sensitivity to tolerances | Library and parameter sweeps | Baseline/variant comparison with measured part data and uncertainty, not just one optimum |
| Explain disagreement with the lab | Residuals, likely loss/timing offsets, model limitations | Numerical engines and diagnostics | Import measured data, fit only declared parameters, show residuals and parameter identifiability |
| Hand an experiment to a colleague | Reproducible model, measurement plan and results | YAML setups and several exports | A self-contained run bundle with calibration fingerprints, detector mappings, units and software version |

The first two audiences to prioritize are a graduate researcher commissioning an SPDC experiment and a researcher designing a small photonic circuit. They need different default workflows, but should share the same model/result/provenance infrastructure.

## 4. Detailed interface and workflow revisions

### U01 — Organize by research task while keeping engines available

The live UI exposes 13 top-level result tabs, including both “Counts” and “Photon counts”, and then another collection of nested tabs. A researcher must already know which implementation corresponds to the desired observable.

Recommended navigation: **Build → Measurement → Results → Compare & export**. Within Measurement offer named recipes: alignment/coupling, source characterization, HOM, polarization correlations/CHSH, photon-number/heralding, and custom analysis. Each recipe selects a compatible default engine and states its assumptions; advanced users can change it.

Keep an engine-based advanced workspace as an alternative. Merely renaming the existing tabs is cheaper but leaves the interpretation problem intact. A complete wizard-only application would constrain expert use too much. Task-oriented defaults with accessible advanced controls are the best compromise.

**Acceptance:** a new user can load a Bell setup, choose polarization correlations, identify four channel pairs, run and export a labelled result without knowing module names or editing YAML.

### U02 — Make the measurement definition a first-class saved object

Separate the optical apparatus from the acquisition: detector/channel mapping, gate width, coincidence full-width convention, relative centers, pulse clock, acquisition duration per setting, number of repeats, accidental estimator, dead-time model and conditioning rules.

Show a compact measurement card above results. Explain the difference between coherence window, repetition period, coincidence window and acquisition duration. Do not reuse “window” without qualification. Changing the measurement must visibly create a new result configuration, not silently reinterpret an old table.

**Acceptance:** reopening a saved experiment reproduces channel mapping and all measurement settings. A comparison with incompatible windows is blocked or explicitly converted with a stated formula.

### U03 — Put assumptions and validity beside the result

Add a compact model summary: simulated source statistics; wavelength/spectral approximation; polarization basis; spatial-overlap treatment; detector response; included timing effects; omitted tail; parameter/calibration status.

Use actionable messages: “This model omits dead time; predicted clicks exceed the detector's recovery regime” is more useful than a hidden generic Notes entry. Avoid advertising the Gaussian tab as universally “exact photon statistics”; it is exact within its implemented Gaussian/source/window assumptions, not an exact model of every apparatus.

**Acceptance:** users can identify whether a displayed rate includes dead time and whether a distribution is truncated without opening source code or documentation.

### U04 — Give results explicit names, units and denominators

Use labels such as “photon arrival rate”, “potential click rate”, “recorded singles”, “coincidences in a 1 ns full-width window”, “excess above accidentals”, “probability per pulse”, and “probability conditioned on herald H”. Rename the existing Counts view to identify its model or observable and distinguish it from circuit outcomes.

Every probability plot should show normalization and omitted mass. A conditional view should always retain heralding success probability and expected accepted-event rate. Display undefined values with a reason. List detector IDs beside compact ket notation; distinguish optical Fock occupations from detector-click vectors.

**Acceptance:** exported columns have enough information to reconstruct their denominator, units, detector order and correction policy without referring to the UI.

### U05 — Add measured-data comparison before more speculative simulation features

Start with CSV import for x, y, uncertainty, units, acquisition time and background/sideband columns. Add an explicit mapping preview and validation. Preserve original data; derived background-corrected data must remain traceable. Overlay model and measurement, show residuals and report the selected objective.

Useful first fits: polarization fringe amplitude/offset/visibility, coincidence peak center/width/background, HOM delay offset/visibility/width, and a loss-budget scale. State which parameters were fitted versus independently calibrated. Do not fit brightness and total detection efficiency freely to singles alone and present a unique physical answer: those quantities can be confounded.

**Acceptance:** a synthetic dataset with known parameters is recovered within stated uncertainty; changing units does not change the fit; a poorly identifiable fit reports that limitation rather than only a best value.

### U06 — Add calibration and uncertainty records to library parts

Generic catalog entries are useful starting points. A lab needs serial number or local ID, measured curve, measurement date, operating conditions, uncertainty/covariance where available, source of values and validity range. Keep vendor nominal data and local measured data distinct.

A result should show which critical values are generic placeholders. Support distributions/tolerances for efficiency, loss, waveplate zero offset, pump power and timing offset. Propagate these separately from counting noise. Begin with controlled Monte Carlo or local sensitivity propagation for a small supported parameter set.

**Acceptance:** changing a calibration invalidates affected results; a run bundle retains its original calibration values even if the library later changes. A user can identify the largest uncertainty contributors.

### U07 — Unify sweeps across supported observables

The current general sweep pipeline evaluates the base outcome; new Fock probabilities are not automatically available merely because Photon counts exists. Add an observable registry whose entries declare backend, units, domain and aggregation/conditioning semantics.

Offer a saved sweep recipe: physical control, bounds, units, selected observable, measurement settings, resolution and run limits. Examples: pump power versus useful heralded counts and g2; phase versus coincidence probability; waveplate angle versus polarization correlation. Support baseline comparison and multiple metrics instead of forcing a researcher to optimize counts while accidentally worsening multipair contamination.

**Acceptance:** the Fock template's phase fringe can be scanned and exported using the same preserved detector/input settings as its single-point calculation. An unsupported delay-dependent Fock scan explains why the rate/temporal model is needed.

### U08 — Make acquisition planning quantitative and honest

Show expected useful events and anticipated statistical precision for an explicitly chosen acquisition model. Allow the user to target a precision or event count and obtain estimated acquisition time, including all measurement settings. Report low-count or rare-herald limitations.

For CHSH, distinguish 10 seconds per setting from 10 seconds divided among settings. For conditional distributions, acquisition time depends on heralding success. Do not use an ideal Poisson precision forecast as a claim that drift/systematics or detector correlations are absent.

**Acceptance:** halving success probability doubles the expected time to collect the same number of accepted events, all else equal. Fixed-pulse uncertainties agree with RR05.

### U09 — Preserve runs and comparisons

Provide immutable run records with names and timestamps. Store setup snapshot, library/calibration fingerprint, measurement recipe, engine/version, seed, source statistics, truncation/resource limits, detector model, warnings, units and normalization. Include the actual data behind plots.

Allow pinning a baseline and comparing a variant without overwriting it. Export a directory or archive containing machine-readable configuration, data tables and a human-readable summary. Preserve undefined values using an explicit machine-readable convention rather than ambiguous JSON NaN/Infinity.

**Acceptance:** a colleague can rerun the bundle without the author's mutable local library. Reports cannot accidentally pair old results with newly edited settings.

### U10 — Make the photon-count workspace readable and editable safely

The live screenshot showed clipped column headings and several large empty tables before running. Collapse component inventory by default; prioritize measurement summary, main observable and model status. Use readable widths, unit-aware headers, copy-selected-cells and clear empty states. Explain “Import from setup” versus “Extract to circuit editor”, including whether edits remain linked to the original canvas.

Use validated integer editors for photon occupations and wire counts. The current editor converts parsed floats with `int(...)`; do not silently turn a fractional occupation into another experiment. Preserve advanced detector fields even when not shown. Ensure ordinary open/save applies to the intended named document.

**Acceptance:** keyboard navigation and a laptop-sized window expose the primary run controls and result interpretation. No physical input is silently rounded/truncated. Opening and saving without edits preserves the complete experiment.

### U11 — Make cancellation and resource limits end-to-end

Before a Fock run, estimate modes, photon sectors, loss/environment modes, mixture branches and output patterns. Extend resource budgeting to sample generation and rendering. The GUI permits up to 100 million shots; `sample()` allocates an index array and a Python sequence for every shot. The existing permanent-work limit does not bound that allocation. This is a source-confirmed risk; it was not stress-tested and is not assigned as the cause of the user's memory problem.

Use multinomial counts when only histograms are needed; generate full sequences only when requested, in bounded chunks with cancellation. Render large distributions with pagination/top-k plus an explicit hidden-mass summary. Preflight should reject obviously excessive requests before computing.

**Acceptance:** large shot requests either remain within a configured budget or fail early with an estimate and remedy. Cancellation is checked during sampling and rendering as well as propagation. Partial/cancelled results are visibly labelled.

### U12 — Add a GUI Bell/polarization workflow

The new CHSH workflow is present in `analysis/chsh.py` and CLI; no CHSH GUI workflow was found in the inspected GUI code. Add four named detector-pair mappings, two settings per arm, a clear distinction between HWP angle and analyzer angle, raw/corrected result views, predicted counts per setting and an exportable acquisition plan.

Keep marginal Stokes/DOP and joint-state information in separate labelled views. An individual photon from a Bell pair can have zero DOP while the pair is entangled. Future tomography should use physical density-matrix reconstruction with uncertainty and documented measurement completeness, rather than treating single-photon ellipse plots as pair-state verification.

**Acceptance:** the Bell template reproduces its expected model correlations through the GUI, angle conventions are explicit, all four setting pairs are visible, and no uncertainty/significance is reported without valid acquisition assumptions.

## 5. Structure changes that support these workflows

The repository already has useful separation between components, engines, analysis, setup/library data and GUI. Preserve that separation. The new `fock` package is a sensible boundary, and reuse of component Jones transfers helps avoid competing optical conventions.

The main structural concern is concentration of responsibilities: `gui/fock_panel.py` has 1,546 lines, `fock/from_setup.py` 1,179, and `engines/rates.py` 1,944 at this revision. Line count is not itself a bug. Here it corresponds to several jobs sharing state, which contributes to the persistence/lifecycle problems above.

Recommended boundaries:

1. **Experiment document:** complete serializable circuit/setup/measurement configuration, stable identity and revision. GUI controls edit it rather than recreate it.
2. **Source specification and preparation:** typed statistical source description; resolve brightness units, pair statistics, tags, spectral assumptions and pulse clock once. Do not pass a mean-power proxy as if it retained a Fock distribution.
3. **Compilation result:** optical transfer, input/output basis map, detector mapping, source mapping, wavelength assumptions and a structured approximation list. “Verified extraction” should declare precisely which objects were compared.
4. **Detector/acquisition model:** distinguish per-window response matrices from temporal dead time/jitter/afterpulse processing. Reuse a response interface while retaining clear capability limits.
5. **Run request and result:** immutable input snapshot plus typed observable records, provenance, status, tail/convergence data and diagnostics. Share this contract across GUI, CLI and exported bundles.
6. **Analysis services:** CHSH, correlations, conditioning, sweeps and uncertainty operate on explicit observables/acquisition definitions rather than untyped dictionary keys and implicit conventions.
7. **Presentation models:** separate circuit editing, setup preparation, output tables, plots, conditions and run orchestration. Keep numerical work out of Qt widgets; keep UI formatting out of engines.

Migrate incrementally, beginning with preserving `Experiment` in the editor and replacing the incomplete `RunGuard` contract. Avoid a broad rewrite before fixing the reproduced defects. Add differential tests at boundaries: document round-trip, compile/extract/run equivalence, response-matrix conservation, backend parity, and observable comparison under matched assumptions.

## 6. Physics validation priorities

- **Photon statistics:** retain vacuum/loss/truncation as distinct concepts; test thermal, Poisson and Fock limits; compare moments only after accounting for tails. Validate high-gain approximations against compatible Gaussian models rather than extrapolating low-pair agreement.
- **Schmidt structure:** the new pair-number model uses equal-weight modes and rounds noninteger K for tag assignment, as documented. K alone does not specify a general spectrum of mode weights. Preserve this approximation explicitly; later accept measured/calculated mode weights where needed.
- **Distinguishability:** tags and scalar overlap are useful models, but spectral-intensity overlap does not include spectral phase or timing. The documented overlap upper bound must remain visible in predictions. Do not present a delay-independent circuit result as a HOM delay model.
- **Polarization:** keep local beam-frame/reflection conventions explicit and tested. Do not “fix” a Jones sign without proving inconsistency. Validate H/V, D/A, R/L basis adapters and both product/Bell inputs through loss and analysis optics.
- **Detector statistics:** validate response on vacuum and at support boundaries, then losses/efficiency/dark counts and multiplexing. Temporal detector effects need separate validation; do not duplicate efficiency in circuit loss and detector response.
- **Cross-engine comparisons:** compare the same observable, source state, time window, normalization, detector response and acquisition regime. Rate, Gaussian, Monte Carlo and Fock outputs can legitimately disagree when those definitions differ. Provide a comparison manifest that explains the differences.
- **Perceval bridge:** retain small analytic reference cases and randomized parity tests. Test actual file/editor round-trips as well as core calculations. A numerically correct core does not protect against parameters lost by the GUI.

## 7. Recommended implementation order and release criteria

### Stage A — preserve scientific meaning

Fix RR01–RR08, then RR11. Add narrowly targeted regression tests using the saved independent probes. Complete the interrupted current-revision suite sequentially on a machine with adequate resources and record the actual environment and result. Do not accept a core test pass as evidence that editor/save/export semantics are preserved.

Release gate: saved experiments preserve their physics; stale results cannot look current; omitted mass is handled explicitly; clocks are not invented; uncertainty and duration semantics agree; dark counts survive vacuum.

### Stage B — make models understandable

Address RR09/RR10 and U02–U04. Add the source/backend capability checks, measurement card, clear result names and visible validity limitations. Make the shipped templates demonstrate valid laboratory regimes or prominently mark deliberate idealizations.

Release gate: a researcher can explain what each plotted quantity measures and which detector/timing effects it includes directly from the result screen.

### Stage C — deliver two complete research recipes

Implement source characterization and photon-number/heralding workflows, with saved runs, cross-engine sweep support and exports. Add the CHSH GUI using the corrected inference model. Preserve advanced engine access.

Release gate: a second researcher can reproduce an exported run, and can compare a baseline/variant without losing inputs or confusing conditional with unconditional rates.

### Stage D — connect to measurements

Add measured-curve import, calibration records, a limited set of validated fits and uncertainty propagation. Expand physical models only in response to a clearly defined measurement discrepancy or experiment need.

Release gate: synthetic-data recovery, uncertainty calibration, unit conversion and identifiability checks pass; the user can trace a prediction back to calibration data and an acquisition recipe.

## 8. Verification result on resumption

The sequential bounded run completed successfully: **23 passed, 7 deselected in 0.74 s**, exit code 0. It selected analytic PNR thermal/product cases, CAR conventions, time-tag clock round-trip and legacy-file checks, delayed-herald peak detection, peak absence, atomic-write failure handling, and detector response checks from `tests/test_revision_engines.py`.

Command:

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/private/tmp/sequoia-round2-mpl \
  venv/bin/python -m pytest -q tests/test_revision_engines.py \
  -k 'pnr_matches_geometric or pnr_two_channel or car_conventions or pulse_clock_survives or reading_version_one or herald_peak_found or peak_not_found or write_failures or detector_response_matches' \
  --basetemp=/private/tmp/sequoia-round2-resume
```

The two interrupted large-suite logs remain incomplete. The 23-test pass verifies the selected fixes; it does not contradict the separate new edge-case reproductions, and it is not a full-suite or exhaustive optical-model validation.
