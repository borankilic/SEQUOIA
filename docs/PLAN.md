# SEQUOIA: Implementation Plan

**SEQUOIA** (se-KWOY-uh): *Simulated Evaluation of QUantum Optics Instruments and Analysis*. Python import name `sequoia`.

## Context

The lab works with CW and pulsed lasers, SPDC sources, free-space and fiber optics, polarization optics (wave plates, paddles), filters, SPAD/SNSPD detectors and time taggers. The goal is a desktop app where an experimental setup is built from a library of **characterized lab parts**. For any point in the setup it predicts the classical beam properties: power, spectrum, polarization, beam size and divergence. At the detectors it predicts the photon-counting observables: singles, coincidences, CAR, heralding efficiency, g2, HOM visibility and polarization correlations. The point is to design and debug setups before and while building them.

The project folder `quantum_experiment_simulation/` is empty, so this is greenfield.

**Decisions from Q&A**
- The canvas is a schematic on a grid. Components can be placed anywhere and rotated. They are connected by **free-space links** or **fiber links**, each carrying a length.
- The app is a Python desktop app (PySide6).
- In the first iteration, data gets into parts by CSV/XLSX import and manual entry. The PDF plot digitizer, AI-assisted datasheet extraction and characterization wizards are deferred to a later iteration (§9).
- v1 must correctly predict four reference experiments:
  1. Heralded HBT with SPDC.
  2. Laser beam shaping and fiber coupling.
  3. Polarization entanglement (fringes, CHSH, tomography).
  4. HOM interference.

---

## 1. Prior art: what to reuse and what to build

| Tool | What it is | Role here |
|---|---|---|
| [spdcalc](https://spdcalc-py.readthedocs.io/) (Rust + Python) | SPDC phase matching, JSA/JSI, singles, coincidences and efficiencies for Gaussian collection modes, HOM | **Dependency.** The physics behind `SPDCCrystal` |
| [thewalrus](https://the-walrus.readthedocs.io/) | Gaussian-state photon statistics (hafnians, torontonians) | **Dependency in M8.** Exact multi-pair and PNR statistics |
| [py-pol](https://py-pol.readthedocs.io/) | Jones, Stokes and Mueller library | **Test oracle** for the polarization code (the runtime code uses vectorized numpy 2×2 per λ) |
| [DCC-Lab raytracing](https://raytracing.readthedocs.io/) | ABCD matrices, Gaussian beams, some Thorlabs lenses | **Test oracle** for the ABCD/Gaussian code |
| [Perceval](https://perceval.quandela.net/) (Quandela), QuTiP | Fock-space linear optics with imperfect sources | Possible later backend for non-Gaussian sources such as quantum dots |
| [SeQUeNCe](https://arxiv.org/abs/2009.12000), [polarization extension (2026)](https://arxiv.org/abs/2604.07289), [NetSquid ZALM sim](https://advanced.onlinelibrary.wiley.com/doi/10.1002/qute.202500810) | Discrete-event quantum network simulators with SPDC, detector and fiber models | Network-scale tools, not table-level optics. Their source and detector models confirm the Monte Carlo approach in M7 |
| [Finesse 3](https://finesse.ifosim.org/) | Classical modal (Hermite-Gauss) interferometer and cavity simulation | Inspiration for adding cavities later |
| [Quantum Flytrap Virtual Lab](https://quantumflytrap.com/virtual-lab) | Drag-and-drop optical table for teaching | UI inspiration only. It does no quantitative modeling of lab parts |
| [Anubuddhi](https://arxiv.org/abs/2512.15736), [PyTheus](https://arxiv.org/abs/2210.09980), [SOQCS](https://www.sciencedirect.com/science/article/pii/S2352711023002996) | LLM-driven design, automated discovery, stochastic circuit simulation | Different goals. Anubuddhi notes that its numerical predictions need expert review, which is the gap this project fills |
| refractiveindex.info DB | Sellmeier and dispersion data | Material data for wave plates, crystals and fibers |

**Gap:** no existing tool combines all of the following:
- a lab parts library backed by characterization data,
- Gaussian-beam, power, spectrum and polarization tracking,
- photon-counting predictions with real detector and time-tagger models,
- a GUI.

This project builds that integration and reuses the physics libraries listed above where they fit.

---

## 2. Core physics architecture: one linear network, two statistics layers

There is no single "photon object that knows everything". Everything between the sources and the detectors (lenses, mirrors, BS, PBS, wave plates, filters, fibers, paddles) is **linear optics**. Linear optics transforms classical field amplitudes and quantum mode operators with the same matrices. That gives the following structure:

1. **Network transfer.** It is computed once. For every route from a source port to a link, detector or probe, it holds:
   - a per-wavelength 2×2 Jones matrix, including loss and phase,
   - the group delay and GDD,
   - the Gaussian-beam ABCD per transverse axis.
2. **Classical layer.** It pushes each source's coherency matrix and Gaussian q through those transfers. The result is power, spectrum, polarization and beam size on every link.
3. **Quantum layer.** It applies the *same* transfers to a quantum source description: the biphoton amplitude in v1, and the covariance matrix in M8. The result is count rates and correlations.
4. **Non-linear parts.** Only three kinds need special code:
   - **sources** (`emit`),
   - **nonlinear crystals**, which take the classical pump and emit a quantum pair source plus the residual pump,
   - **detectors** (`detect`), which turn the arriving light into click rates or photocurrent.

```
Setup YAML ─► graph + validation ─► classical pass (lasers → every link)
                                        │ pump state at crystal
                                        ▼
                          SPDC/entangled source → quantum contributions
                                        │ same path transfers
                                        ▼
               detectors + time tagger ─► engines: rates (v1) | Monte-Carlo tags (M7) | Gaussian (M8)
                                        ▼
                          readouts, probes, diagnostics, sweeps
```

### 2.1 The light object

```python
Spectrum       # 'line' (center, FWHM, shape gauss/lorentz/sech2)  or  sampled (λ grid, S(λ))
GaussianMode   # q_x, q_y, M2_x, M2_y, n  → w, R, z_R, divergence, distance to waist
Path           # one route from a source to here
  jones: (Nλ,2,2) complex      # accumulated amplitude transfer: loss, polarization, phase
  group_delay, gdd, length     # timing, pulse broadening, layout
  mode: GaussianMode           # spatial mode along this route (ABCD; reset by fiber mode filters)
  history: list[component_id]  # for budgets & diagnostics
Contribution   # light from ONE source on this link: source ref + list[Path]
BeamState      # all contributions on a link + cached derived quantities
```

- **Combining contributions.** Light from different sources adds incoherently. Paths from the same source add coherently:
  `W = Σ_pp' J_p W_src J_p'† · γ(Δτ_pp') · overlap(mode_p, mode_p')`.
  Interferometer fringes, and the visibility lost to coherence length and mode mismatch, come out of this without special-casing. Sampled spectra multiply cross terms by the bin-averaging factor `sinc(πΔν Δτ)` so that coarse grids stay correct.
- **Derived quantities on each link:**
  - power and photon flux,
  - mean photon number per pulse (pulsed) or per coherence time,
  - spectrum,
  - Stokes vector, DOP and ellipse,
  - w_x, w_y, R, divergence and distance to the waist,
  - a statistics tag: coherent, thermal or pair.
- **Quantum contributions.** The signal and idler of a pair source carry the same Path data. The value displayed on a link is photons/s = pair rate × path efficiency.
- **Spatial overlap.** The overlap between a beam and a fiber mode is computed in the same plane:
  `η = 4 / [(w1/w2 + w2/w1)² + (πw1w2/λ)²(1/R1 − 1/R2)²]`.
  An optional misalignment term is added only when the user sets one.

### 2.2 Quantum layer v1: rate engine

The rate engine works to first order in the pair rate (low gain). That is the standard regime for CW and weakly pumped SPDC, and it is exact for single pairs.

- **Source.** The source is described by:
  - a biphoton amplitude `F_{σσ'}(ω1,ω2)`, which combines the JSA with the polarization indices and comes from spdcalc or an analytic form,
  - a pair rate R in pairs/s.

  A mixed polarization state is represented as an ensemble of its eigenstates.
- **Coincidence formula.** One formula covers heralding, HBT, HOM (through the exchange term), Bell tests and tomography (through the polarization indices), and Franson (through the path sums):
  `A_AB(ω1,ω2) = J_{A←s}(ω1)⊗J_{B←i}(ω2)·F(ω1,ω2) + J_{A←i}(ω1)⊗J_{B←s}(ω2)·F(ω2,ω1)`
  `P_AB = ∫∫ A†(D_A⊗D_B)A`
  Here path delays enter as phases, and D is the detector POVM (an efficiency η(λ) that depends on polarization).
- **Singles.** Singles = R·η_marginal + dark counts + classical leakage. Residual pump × filter OD × detector QE is one example of leakage. A non-paralyzable dead-time correction is applied.
- **Coincidence window.** The window factor is the fraction of the correlation peak that falls inside the coincidence window:
  - The peak shape is the biphoton correlation time convolved with the detector and tagger jitters.
  - The peak position comes from the path delays plus the cable delays.
- **Accidentals.** For CW, accidentals are `S_A S_B τ_w`. Pulsed sources use the per-pulse formula. Multi-pair triple terms feed the heralded g2(0).
- **Unheralded g2(τ).** It comes from the reduced spectral density matrix of the filtered JSA, which gives 1+1/K at τ=0. The result is convolved with the jitter, so the simulator correctly predicts when bunching becomes invisible with 50 ps detectors.
- **Beyond v1:**
  - The Gaussian engine (M8) handles exact multi-pair statistics, high gain and PNR detectors.
  - The Monte Carlo time-tag engine (M7) handles dead time, afterpulsing and real histogram shapes.

### 2.3 Conventions and v1 assumptions (`docs/conventions.md`, enforced by tests)

- **Units.** Values are SI internally. Units are parsed with pint only at the boundaries (UI and files).
- **Polarization frame.** H lies in the table plane and V is the table normal.
  - An ideal mirror has Jones matrix diag(1, −1), which flips handedness.
  - A metal or dielectric mirror adds an s-p retardance taken from coating data.
  - The BS is symmetric (r = i√R).
- **Spatial modes.** Optics are paraxial with one Gaussian transverse mode per path (embedded Gaussian with M²). Alignment is ideal unless the user enters misalignment parameters.
- **Topology.** Only feed-forward networks are supported: no cavities and no back-reflections. A cycle is a validation error.
- **Physics simplifications.** The pump is undepleted, components are stationary, and detectors are threshold detectors until PNR arrives in M8.
- **Fiber-link polarization.** For SMF links the default transform is the identity, which is the ideal case. Alternatives are an explicit SU(2) or a seeded random one. Paddles can then be optimized against it.

---

## 3. Component model: category → type → part → instance

- **Category.** A palette group that also defines the port conventions. The categories are Sources, Nonlinear, Free-space optics, Polarization, Filters, Fiber, Detectors, Electronics and Probes.
- **Type.** A Python class that holds the physics and a parameter schema (Pydantic). Example: `SNSPD`.
- **Part.** Data only, no code: a folder with `part.yaml`, CSV files and the original datasheet or raw files. It fills the schema for one product or one physical unit. Example: "Single Quantum Eos, ch2, S/N …".
- **Instance.** A part placed in a setup, plus per-instance settings (angle, temperature, length, bias) and optional overrides.

**Parameter kinds:**
- `Scalar` with a unit,
- `Curve` (x → y), for example QE(λ),
- `Surface` (x, y → z), for example SDE(λ, I_bias) or brightness(P, T),
- `Enum`,
- `File`.

Every value carries **provenance**: datasheet, raw-data, measured, model or ideal-default. The kinds `digitized` and `ai-extracted` are added in iteration 2.

**Resolution order:** instance override > measured > datasheet/raw-data > physics model > ideal default.

There is a global **Ideal ↔ Characterized** toggle, and each component can be set to "treat as ideal". Together they give a loss and noise budget that shows which component limits the result.

### v1 types

| Category | v1 types | Key parameters (*curves in italics*) |
|---|---|---|
| Sources | `Laser` (CW / pulsed) | λ0, *spectrum* or linewidth, power or *P(I)*, polarization + PER, w0/divergence, M², rep rate, pulse width/shape |
| Nonlinear | `SPDCCrystal`, `EntangledPairSource` | material, PM type, poling period, length, T, d_eff, collinear or non-collinear; measured brightness (pairs/s/mW) or *brightness(P,T)*; Bell state, visibility, phase |
| Free-space | `Lens`, `Objective`, `Mirror` (flat/curved), `BeamSplitter`, `DichroicMirror`, `Iris`, `BeamDump` | f or *f(λ)*; EFL/NA/WD/*T(λ)*; R_curv, AOI astigmatism; *R(λ),T(λ)* per polarization; aperture |
| Polarization | `WavePlate`, `Polarizer`, `PBS`, `FiberPaddleController` | *retardance(λ)* or material + order, angle; extinction; *T_p(λ), R_s(λ)*; loops, loop and cladding diameters, paddle angles (bend-birefringence model) |
| Filters | `SpectralFilter` (BP/LP/SP/notch), `NDFilter` | *T(λ)* or *OD(λ)*, AOI; OD |
| Fiber | `OpticalFiber` (an SMF/PM/MMF fiber *type* that fiber links reference), `FiberCoupler` (collimator / FiberPort), `FiberBeamSplitter` | *MFD(λ)*, NA, cutoff, *attenuation(λ)*, dispersion, PER; f, fiber type; *split(λ)* |
| Detectors | `SPAD`, `SNSPD`, `Photodiode` | *PDE(λ)*, DCR, dead time, afterpulsing, jitter (FWHM + tail), active area, *saturation correction*; SNSPD: polarization extinction, *SDE(λ, I_b)*; PD: *responsivity(λ)*, dark current, NEP, gain |
| Electronics | `TimeTagger`, `Cable` | channels, jitter, dead time, input delays; coincidence groups and windows; cable delay |
| Probes | `PowerMeter`, `BeamProfiler`, `Polarimeter`, `Spectrometer` | non-invasive; attach to any link |

### Links

Each port is typed free-space, fiber or electrical. There is one link kind per port kind:
- **Free-space link** between free-space ports. It holds a length and a medium. The length is typed in, or can optionally be locked to the distance on the canvas using a grid scale (default 1 cell = 25 mm, matching breadboard holes).
- **Fiber link** between fiber ports. It references an `OpticalFiber` part and holds a length, a connector type and loss, and a polarization transform. Attenuation, dispersion, multimode warnings and delay come from the fiber part.
- **Cable** from a detector output to a time-tagger channel. It holds a delay.

Validation enforces these kinds. A free-space beam cannot enter a fiber without a `FiberCoupler`, and a detector output must go to a tagger channel.

**Collinear SPDC.** It has one output port that carries the residual pump (classical), the signal and the idler. Downstream dichroics, filters and PBSs separate them physically, so pump leakage is modeled for free.

### Adding things

- **New part (the common case), no code.** Library Manager → choose a type → a form generated from the schema → fill it in and import curves → save. The part appears in the palette.
  - *Example, Thorlabs photodiode:*
    1. Type `Photodiode`.
    2. Import the responsivity xlsx and map Wavelength (nm) → Responsivity (A/W).
    3. Enter the active area, dark current, NEP and bandwidth.
    4. Attach the PDF.
    5. Save.
- **New type.** One Python file in `components/<category>/`:
  1. Subclass `Component`.
  2. Define a `Params` model.
  3. Implement `transfer()` for linear parts, or `emit()` / `detect()`.
  4. Decorate with `@register("filters", "Etalon")`.

  The GUI forms are generated from the schema, so no GUI code is needed.

**Part file sketch** (`library/detectors/snspd/sq_eos_ch2/part.yaml`):
```yaml
type: SNSPD
manufacturer: Single Quantum
model: Eos
serial: "…"
params:
  efficiency: {curve: sde_vs_wavelength.csv, x: wavelength (nm), y: efficiency (1),
               provenance: {source: measured, date: 2026-03-01, by: Boran, note: "Ib=12 uA"}}
  polarization_extinction: {value: 3.0, unit: "1", provenance: {source: datasheet}}
  dark_count_rate: {value: 50, unit: Hz}
  dead_time: {value: 20, unit: ns}
  jitter_fwhm: {value: 15, unit: ps}
attachments: [datasheet.pdf]
```
Setup files (`setups/*.yaml`) list the instances (part ref, settings, canvas position and rotation), the links (free-space: length and medium; fiber: fiber part, length and connector loss; cable: delay), the probes and the analyses (coincidence groups, windows, sweeps).

---

## 4. Getting datasheet and characterization data in (first iteration)

1. **Raw-data files.** A CSV/TSV/XLSX importer:
   - detects the header row,
   - lets the user pick columns,
   - parses units from headers such as `(nm)`, `A/W`, `%`, `OD` or `dB`,
   - converts the y-kind (% / OD / dB → T; QE ↔ responsivity via R = ηqλ/hc),
   - shows a preview plot,
   - runs sanity checks (monotonic x, 0 ≤ T ≤ 1).

   Thorlabs graphs usually come with a "raw data" xlsx. Semrock and Chroma provide ASCII spectra. The lab's own measured curves, such as a filter transmission or an SNSPD efficiency curve, are imported the same way and tagged `measured`.
2. **Manual entry.** Scalars are typed into the schema form. Short curves can be typed into an editable table. The provenance (datasheet page, measurement date, who) is entered per field.

The original PDFs and raw files are stored in the part folder so every value can be traced. Is a spec sheet alone enough? No: an attached PDF is documentation only. Its numbers and curves have to be entered or imported into the schema. Typical-value datasheet curves also rank below the lab's own measurements in the resolution order.

Three features are deferred to a later iteration (§9): the PDF plot digitizer, AI-assisted extraction through the Claude API, and the characterization wizards.

---

## 5. GUI (PySide6)

- **Main window:**
  - **Palette dock:** a category tree plus search over library parts.
  - **Schematic canvas:** a custom `QGraphicsScene` on a grid with:
    - placement anywhere, with toggleable snap-to-grid,
    - rotation in 90° steps by shortcut, or free rotation with 15° snapping, plus flip,
    - fixed-port component items whose ports rotate with the component,
    - two link tools, each labeled with its length:
      - **free-space**, drawn as a straight beam line colored by wavelength,
      - **fiber**, drawn as a curved cable,
    - refusal of links between incompatible ports,
    - electrical cables from detectors to tagger channels,
    - undo/redo via `QUndoStack`, copy and paste.

    Rotation on the canvas is layout only. Optical angles, such as a wave plate's axis or a mirror's AOI, are parameters in the inspector.
  - **Inspector dock:** a form generated from the Pydantic schema, with quantity spinbox + unit, enums, angle slider and curve preview.
  - **Results dock:** tabs, listed below.

  The canvas is custom rather than NodeGraphQt or a similar framework because links must carry physical lengths, ports are typed optical or electrical, and links need hover beam cards. That takes a few hundred lines and avoids fighting a dataflow-node framework.
- **Live updates:** the classical and rate engines rerun after every edit, debounced to about 150 ms, in a `QThreadPool` worker. The Monte Carlo and Gaussian engines run on demand with a progress bar.
- **Link interaction:**
  - Hovering over a link shows a beam card: P, flux, w_x/w_y, R, distance to the waist, polarization ellipse, DOP and a spectrum sparkline.
  - Right-clicking a link adds a probe.
- **Results tabs:**
  - **Diagnostics.** Clicking a warning selects the component.
  - **Beam caustic.** w(z) along a chosen path, with component markers.
  - **Efficiency/power budget waterfall.**
  - **Polarization.** Ellipse and Poincaré sphere.
  - **Spectra.**
  - **Counts.** A table of singles, coincidences, accidentals, CAR, η_h and g2.
  - **Histograms.** g2(τ), HOM dip and fringes.
  - **Sweeps.**
- **Sweeps and optimizer:**
  - Sweeps are 1D or 2D over any numeric instance parameter, including link lengths.
  - The optimizer uses scipy. Examples: lens positions → maximum fiber coupling; paddle angles → maximum SNSPD efficiency.
- **Library Manager window:**
  - browse, create, duplicate and edit parts,
  - curve fields with Import (CSV/XLSX) / Edit table,
  - provenance shown per field.
- **Diagnostics checked:**
  - aperture clipping,
  - unterminated output ports (lost light),
  - a fiber running multimode below its cutoff,
  - curve extrapolation outside the data range,
  - detector saturation or dead-time loss above a threshold,
  - pump-leakage count rate at detectors,
  - low coupling efficiency,
  - a polarization-sensitive detector that is misaligned,
  - a coincidence peak outside the window (delay mismatch).
- **Plots:** pyqtgraph, with PNG and CSV export.

---

## 6. Repository layout

```
quantum_experiment_simulation/
  pyproject.toml            # uv, src layout, package sequoia, extra: [gaussian]
  src/sequoia/
    units.py  data.py       # pint boundary parsing; Scalar/Curve/Surface/Provenance
    light/                  # spectrum.py, gaussian.py, state.py (Path, Contribution, BeamState)
    components/base.py      # Component, PortSpec, PortTransfer, registry
    components/{sources,nonlinear,freespace,polarization,filters,fiber,detectors,electronics,probes}/
    materials/              # Sellmeier data (quartz, KTP, BBO, LN, silica) + loader
    setup/                  # model.py (Setup/Instance/FreeSpaceLink/FiberLink/Cable), io.py (YAML), validate.py
    engines/                # network.py, classical.py, rates.py, montecarlo.py (M7), gaussian.py (M8)
    analysis/               # metrics.py (CAR, η_h, g2, CHSH, tomography), correlation.py, sweep.py, optimize.py
    library/                # store.py, importers/table.py
    gui/                    # app.py, main_window.py, canvas/, inspector/, results/, library_manager/, dialogs/
    cli.py                  # `sequoia` (GUI) and `sequoia run setup.yaml [--engine rates|mc]`
  library/                  # the lab's part library (git-tracked data); path configurable
  setups/                   # laser_fiber_coupling, mzi, hbt_heralded, bell_test, hom (.yaml)
  tests/                    # unit, reference setups, cross-engine, gui (pytest-qt)
  docs/conventions.md
```

**Tech stack:**
- Python 3.12 with uv,
- numpy, scipy, pint, pydantic v2, ruamel.yaml, pandas + openpyxl, networkx (topological sort and cycle detection),
- spdcalc,
- PySide6, pyqtgraph,
- thewalrus (`[gaussian]` extra),
- dev tools: pytest, pytest-qt, ruff.

The library defaults to a git-tracked folder. It can later be pointed at a shared lab repo or drive.

---

## 7. Milestones

The physics comes first and stays headless, so it is testable. The GUI comes after.

**M0: Scaffolding**
- Tasks:
  - `git init`, then `uv init --package sequoia` with the src layout. Check PyPI for a name clash before ever publishing (the distribution name can differ, e.g. `sequoia-optics`, while the import name stays `sequoia`).
  - Add the dependencies.
  - Configure ruff and pytest.
  - Add `.gitignore`.
  - Write the `docs/conventions.md` skeleton.
- Done when:
  - `uv run pytest` is green.
  - `uv run sequoia` opens an empty main window.

**M1: Data model, part library, setup files**
- Tasks:
  - `units.py` and `data.py`: Curve/Surface interpolation with an explicit extrapolation policy (warn + clamp by default), and Provenance.
  - `components/base.py`: the registry.
  - `library/store.py`: part folder load/save and the index.
  - `setup/`: the model (instances with canvas position and rotation; free-space links, fiber links, cables), YAML I/O, and validation (port and link kinds, dangling ports, cycles).
  - `library/importers/table.py`: the CSV/XLSX → Curve importer with y-kind conversions.
- Done when:
  - Parts and setups round-trip through YAML.
  - A %-transmission CSV, an OD CSV and an xlsx of the same filter import to identical SI curves.
  - Invalid setups raise clear errors.

**M2: Classical engine + linear components** (target: laser beam and fiber coupling)
- Tasks:
  - `light/` and `engines/network.py`: path enumeration, coherent combination, and a BeamState per link.
  - Every linear type from the table.
  - Fiber links: mode filtering, attenuation, dispersion and polarization transform.
  - `Laser`, the probes and the `Photodiode` readout.
  - Diagnostics.
  - Example setups:
    - `laser_fiber_coupling.yaml`: laser → 2-lens telescope → mirrors → fiber coupler → SMF fiber link → power meter.
    - `mzi.yaml`.
- Done when:
  - The analytic tests in §8 pass.
  - `sequoia run setups/laser_fiber_coupling.yaml` prints the link table and the coupling efficiency.

**M3: Quantum sources + rate engine** (targets: HBT, Bell, HOM)
- Tasks:
  - `SPDCCrystal` on top of spdcalc: λs/λi, JSA, temperature tuning, efficiencies, HOM.
  - **Collection-mode back-propagation:** take each fiber's mode backward through the ABCD chain to the crystal, which gives spdcalc the signal and idler waists and positions. spdcalc then returns the fiber-coupled pair rate and heralding efficiencies. A measured brightness overrides the absolute rate.
  - `EntangledPairSource` (Sagnac or crossed-crystal black box).
  - `SPAD`, `SNSPD`, `TimeTagger` and `Cable`.
  - `engines/rates.py` (§2.2).
  - `analysis/metrics.py`: CAR, η_h, heralded and unheralded g2, visibility, CHSH, and predicted tomography counts with MLE reconstruction under Poisson noise for a given integration time.
  - Example setups:
    - `hbt_heralded.yaml`: 405 nm CW → type-II PPKTP → LP + BP filters → PBS → fiber couplers → herald SPAD; signal → fiber BS → 2 SPADs → tagger.
    - `bell_test.yaml`: Sagnac source → HWP/QWP/PBS per arm → paddles → SNSPDs.
    - `hom.yaml`: type-II pair → PBS → HWP → delay → fiber BS.
- Done when:
  - The §8 checks pass.
  - The unfiltered HOM curve matches spdcalc's `hom_rate_series`.
  - Removing the LP filter in the HBT setup triggers the pump-leakage diagnostic with a sensible count rate.

**M4: GUI shell + schematic editor**
- Tasks:
  - The main window and docks.
  - The palette populated from the library.
  - The canvas from §5: grid placement and rotation, typed ports, free-space and fiber link tools with editable lengths, cables, undo/redo.
  - The schema-driven inspector.
  - Open and save YAML.
  - The debounced live recompute in a worker thread.
- Done when:
  - `hbt_heralded` can be built from scratch in the GUI, and saving it gives a YAML equivalent to the example.
  - The pytest-qt smoke test passes.

**M5: Results, budgets, sweeps, optimizer**
- Tasks:
  - Link hover cards and wavelength coloring.
  - The diagnostics panel.
  - All the results tabs from §5.
  - The ideal-toggle loss/noise budget.
  - `analysis/sweep.py` and `optimize.py`, with their GUI dialogs.
- Done when:
  - Sweeping a HWP in `bell_test` gives the expected fringe and S ≈ 2√2·V.
  - Optimizing the lens distances in `laser_fiber_coupling` with ideal parts reaches η ≥ 0.95.

**M6: Library Manager + data import**
- Tasks:
  - The part browser and editor, with provenance per field.
  - The CSV/XLSX import dialog and the editable curve table.
  - A seed library: ideal generic parts for every type, plus the lab's real parts. The user supplies the vendor raw-data files and datasheets, or approves downloading them.
- Done when:
  - A Thorlabs photodiode part can be created from its raw-data xlsx in under 2 minutes. It appears in the palette and gives the expected photocurrent in a setup.
  - A part created in the GUI round-trips to the same `part.yaml` + CSV as one written by hand.

**M7: Monte Carlo time-tag engine**
- Tasks:
  - Sample pair emission: Poisson for CW, per-pulse for pulsed.
  - Sample each pair's joint outcome from the rate engine's probabilities, which already include interference and polarization.
  - Add jitter (Gaussian + exponential tail), dead time, afterpulsing, dark counts, multi-pair events, classical background, and tagger delays and jitter.
  - Produce per-channel time-tag arrays.
  - `analysis/correlation.py`: coincidences, g2(τ) and heralded g2 from triples. It is written so it can later be pointed at real tagger exports.
- Done when:
  - MC agrees with the rate engine within 3σ for HBT, Bell and HOM at low rates.
  - The dead-time saturation curve follows the non-paralyzable formula.

**M8: Gaussian multimode quantum engine**
- Tasks:
  - Build a covariance matrix over spectral bins and polarization modes from the filtered JSA (Takagi/Bloch-Messiah).
  - Apply losses and linear optics from the same network transfers.
  - Compute closed-form threshold-click probabilities, and PNR probabilities via thewalrus.
  - Cover pulsed and high-gain sources.
- Done when:
  - It matches the rate engine at low gain.
  - It reproduces the known rise of heralded g2(0) and fall of CAR with pump power.

---

## 8. Verification

- **Analytic unit tests:**
  - Malus's law.
  - A HWP at θ rotates linear polarization by 2θ.
  - A QWP at 45° gives circular polarization.
  - PBS extinction.
  - The paddle retardance formula agrees with the Thorlabs FPC numbers.
  - Thin-lens waist transformation.
  - Telescope magnification.
  - Rayleigh range.
  - Curved-mirror AOI astigmatism.
  - The Gaussian overlap formula, with η = 1 for a matched beam.
  - A fiber below cutoff raises a warning.
  - MZI visibility vs OPD = |γ(τ)| of the spectrum.
  - Without filtering, C = R·η_s·η_i and CAR ≈ C/(S_s S_i τ_w).
  - CHSH S = 2√2 for |Φ+⟩ at the optimal angles.
  - HOM visibility = 1 for a symmetric JSA on a 50:50 BS.
  - Unheralded g2(0) = 1+1/K.
  - A coherent laser gives g2 = 1.
- **Oracles:** py-pol (polarization), raytracing (ABCD), spdcalc (HOM, efficiencies).
- **Reference setups:** each `setups/*.yaml` has expected numbers derived by hand in `tests/reference/`. They are run with `sequoia run …` and with pytest.
- **Cross-engine checks:** rates vs MC (M7) vs Gaussian (M8).
- **Against the lab:** one real dataset, such as an HBT run with pump power, singles, coincidences and the part list, run after M3 to compare the prediction with the measurement.
- **GUI:**
  - pytest-qt tests that place, rotate and connect components with free-space and fiber links, then save and reload to identical YAML,
  - a manual run of building HBT from scratch, with screenshots via `QWidget.grab()`.

---

## 9. Out of scope for v1 (extension points kept open)

- **Deferred to iteration 2:**
  - The PDF/image plot digitizer.
  - AI-assisted datasheet extraction through the Claude API, with per-field review.
  - Characterization wizards, e.g. SPDC power sweep → brightness, Klyshko efficiencies, dark counts and the multi-pair term.
- Non-Gaussian sources such as quantum dots, via a Perceval or QuTiP backend.
- Cavities and back-reflections, via a steady-state linear solve like Finesse's.
- Composite or macro components (sub-graphs).
- A 2D optical-table view of the same graph.
- SHG, SFG and FWM.
- Time-dependent elements such as EOMs and AOMs.
- Importing real time-tag files for side-by-side comparison.
- PyInstaller packaging.
