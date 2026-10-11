# SEQUOIA

**S**imulated **E**valuation of **QU**antum **O**ptics **I**nstruments and **A**nalysis (pronounced se-KWOY-uh).

A desktop simulator for quantum optics lab setups. Build an experiment from a library of
characterized parts (lasers, SPDC sources, lenses, wave plates, filters, fibers, SPADs/SNSPDs, time
tagger), then read off what the light does at every link and what the detectors will see:
power, polarization, beam size, singles, coincidences, CAR, heralding efficiency, g2, HOM
visibility and CHSH.

![SEQUOIA main window with the heralded HBT example](docs/img/gui_hbt.png)

One-page overview: [docs/overview.pdf](docs/overview.pdf)

## What it does

- **Schematic editor.** Place parts anywhere on a grid, rotate and flip them, and connect them with
  free-space, fiber or cable links that carry a length. Undo/redo, copy/paste, auto layout.
- **Classical engine.** Per-wavelength Jones matrices and Gaussian beams (ABCD, M²) with coherent
  combination of paths, so interferometer fringes, coherence length and mode mismatch come out
  naturally. Fiber coupling is computed as a mode overlap.
- **Quantum rate engine.** SPDC pair sources (spdcalc) or entangled-pair black boxes, pushed
  through the same network and into detector, dead-time, jitter and coincidence-window models.
- **Parts as data.** A part is a `part.yaml` plus CSV curves with provenance per value
  (datasheet, measured, model, ideal default). CSV/XLSX import converts %, OD, dB and
  responsivity to SI curves.
- **Results and analysis.** Hover a link for its beam card (power, flux, w_x/w_y, R, waist,
  polarization ellipse, spectrum); link colours follow wavelength. Tabs for beam caustic,
  polarization (ellipse and Poincaré sphere), spectra, g2 / HOM histograms, power and pair-photon
  budgets, and 1D/2D parameter sweeps with CSV/PNG export.
- **Optimizer and budgets.** Maximize or minimize any result by tuning bounded parameters
  (scipy), and rank which part limits a result by replacing each part with its ideal.
- **Diagnostics.** Pump leakage, aperture clipping, fiber below cutoff, curve extrapolation,
  coincidence peak outside the window, and more; each links back to the offending component.

```
parts (YAML + CSV) ─► setup graph ─► validation ─► classical engine ─► rate engine ─► readouts
```

## Components

| Category | Types |
|---|---|
| Sources | `Laser` (CW / pulsed) |
| Nonlinear | `SPDCCrystal`, `EntangledPairSource` |
| Free-space | `Lens`, `Objective`, `Mirror`, `BeamSplitter`, `DichroicMirror`, `Iris`, `BeamDump` |
| Polarization | `WavePlate`, `Polarizer`, `PBS`, `FiberPaddleController` |
| Filters | `SpectralFilter`, `NDFilter` |
| Fiber | `OpticalFiber`, `FiberCoupler`, `FiberBeamSplitter` |
| Detectors | `SPAD`, `SNSPD`, `Photodiode` |
| Electronics | `TimeTagger` |
| Probes | `PowerMeter`, `BeamProfiler`, `Polarimeter`, `Spectrometer` |

The shipped `library/` parts are generic placeholders with ideal or model values and no vendor
numbers. Replace them with your measured parts before trusting absolute numbers.

## Example setups

| File | What it shows |
|---|---|
| `setups/laser_fiber_coupling.yaml` | laser, telescope, mirrors, fiber coupler, SMF: coupling efficiency |
| `setups/mzi.yaml` | Mach-Zehnder fringes and coherence-limited visibility |
| `setups/hbt_heralded.yaml` | type-II PPKTP, heralded HBT: singles, CAR, heralded g2 |
| `setups/hom.yaml` | HOM dip versus delay |
| `setups/bell_test.yaml` | polarization correlations and CHSH |

## Project layout

```
src/sequoia/
  units.py data.py        pint boundary parsing, Scalar/Curve/Surface/Provenance
  components/             one module per category; registry of types
  light/                  spectra, Gaussian modes, beam state, pair amplitudes, detection
  engines/                classical.py, rates.py, cache and report helpers
  analysis/               CAR, heralding, visibility, CHSH, tomography (MLE)
  library/                part store and CSV/XLSX importer
  setup/                  setup model, YAML I/O, validation
  gui/                    document, canvas, palette, inspector, results, main window
library/                  part library (data)
setups/                   example setups
docs/                     plan, conventions, M3 report, overview
```

## Status

Milestones M0 to M5 are done: data model and library, classical engine, quantum rate engine, the
GUI shell with the schematic editor, and results plots, sweeps, optimizer and budgets (see
[docs/PLAN.md](docs/PLAN.md)). The test suite covers
analytic checks (Malus, HWP rotation, thin lens, Gaussian overlap, MZI visibility, CHSH
S = 2√2·V, HOM against spdcalc, dead time against Monte Carlo).

Planned: library manager (M6), Monte Carlo time tags (M7), Gaussian multimode engine (M8).

Known limits:

- The absolute SPDC pair rate is uncertain by a factor of about 0.4 to 2; set a measured
  `brightness` on the part for real experiments (details in [docs/M3_REPORT.md](docs/M3_REPORT.md)).
- Collection modes are taken from part data, not back-propagated through the optics.
- Thin lenses, mode tracked at the central wavelength, no GDD or PMD.
- Feed-forward networks only: no cavities or back-reflections.
- The GUI has been exercised through offscreen tests and rendered screenshots; keyboard
  shortcuts have had little live-display testing.

## Setup

```bash
uv sync --all-extras
```

On macOS, a `.venv` directory inside a synced folder (such as Desktop) can get its files flagged `hidden`.
Python then ignores the editable-install `.pth` file and Qt can't find its plugins. This repo therefore
keeps the environment in `venv/` with `.venv` as a symlink to it. If `uv sync` ever recreates a real
`.venv`, remove it and rerun:

```bash
rm -rf .venv && UV_PROJECT_ENVIRONMENT=venv uv sync --all-extras && ln -s venv .venv
```

## Run

```bash
uv run sequoia
```

Open a setup file in the editor (parts come from `./library`, or from a `library/` folder next to the
file or in one of its parents; override with `--library DIR`, or use File > Open Library...):

```bash
uv run sequoia setups/hbt_heralded.yaml
```

Drag parts from the palette onto the canvas (or double-click them), drag from a port to another
port to link them, and edit values in the inspector. The setup is re-simulated in the background
after every edit (Run > Auto-run) and the results dock shows diagnostics, links, readouts and
counts; clicking a diagnostic selects the offending component or link. Shortcuts: `R`/`Shift+R`
rotate, `F` flip, `Delete`, `Ctrl+C`/`Ctrl+V`, `Ctrl+Z`/`Ctrl+Shift+Z`, `Ctrl+L` auto layout,
`Ctrl+G` snap to grid, `Ctrl+=`/`Ctrl+-`/`Ctrl+0` zoom, `Ctrl+Return` run now, `Space`+drag or middle
mouse to pan. Window layout and recent files are remembered between sessions.

Simulate a setup from the command line:

```bash
uv run sequoia run setups/laser_fiber_coupling.yaml
```

The parts come from `library/` (override with `--library DIR`).

## Photon-pair experiments (M3)

Three example setups use the quantum layer (type-II PPKTP SPDC, an entangled-pair black box, SPAD
and SNSPD detectors, a time tagger). The parts in `library/` are generic placeholders; every value
is tagged `ideal-default` or `model` (no vendor numbers), so replace them with measured parts.

```bash
uv run sequoia run setups/hbt_heralded.yaml   # heralded HBT: singles, coincidences, CAR, heralding, g2
uv run sequoia run setups/hom.yaml            # HOM interference; vary the length of link delay_arm
uv run sequoia run setups/bell_test.yaml      # four-outcome polarization correlations, CHSH
```

`sequoia run` prints the classical link table and then a "Photon rates" section: per tagger channel
singles, pair and background counts, coincidence rates for every channel pair, the analyses listed
under `analyses:` in the setup (`coincidences`, `heralded_g2`, `g2`, `histogram`) and rate
diagnostics (`pump_leakage`, `delay_aliasing`, `coincidence_outside_window`, ...). Use `window:`
and `center: auto` (the correlation peak) or an explicit delay that includes cable delays.

Python access:

```python
from sequoia.engines.rates import simulate_rates
from sequoia.analysis import metrics  # car, chsh_s, tomography_settings, mle_density_matrix, ...

result = simulate_rates(setup, library)
result.analyses["herald_a"]["car"], result.channels["tagger.ch1"]["singles_hz"]
```

## Sweeps, optimization and budgets

In the GUI use the Sweeps tab (Run > Sweep parameter... / Optimize parameters...) or the Budget
tab. From Python:

```python
from sequoia.analysis.sweep import linspace_axis, run_sweep
from sequoia.analysis.optimize import Variable, optimize
from sequoia.analysis.budget import ideal_budget

axis = linspace_axis("settings:hwp_a.angle", 0, 90, 37, "deg")
fringe = run_sweep(setup, library, [axis], ["analysis:tt.true_hz"])

best = optimize(
    setup, library,
    [Variable("link:s2.length", 100, 250, "mm")],
    "link:f1.coupling_efficiency",
)
budget = ideal_budget(setup, library, "analysis:herald_a.true_hz")
```

Parameter targets are `settings:<id>.<name>`, `overrides:<id>.<name>` or `link:<id>.<field>`
(`length`, `delay`, `connector_loss_db`); outputs are flat names such as `link:f1.power_W`,
`readout:pd.photocurrent_A`, `analysis:herald_a.car` (see `sequoia.analysis.sweep.metrics_of`).
The ideal budget replaces one part at a time with an ideal version (unit transmission, unit
detector efficiency, no dark counts); it does not touch collection modes or alignment.

### Units and the meaning of `brightness`

Every physical quantity in a part or setup file is written with its unit (`810 nm`, `10 mW`,
`45 deg`, `25 degC`, `100 cps`, `3.9 pm/V`) and is dimension-checked with a message that names the
field. A bare number is only accepted for dimensionless quantities (extinction ratios, split ratios,
optical density, wave-plate retardance in waves, visibility). In particular a bare `temperature: 25`,
`angle: 45` or `dark_count_rate: 100` is an error with a hint (write `25 degC`, `45 deg`,
`100 Hz`; a bare `0` is accepted for an angle), and `-300 degC` (below absolute zero), `25 mW` for
a temperature or `45 mm` for an angle are rejected. Angles are stored in radians internally and
temperatures in kelvin.

`brightness` (on `SPDCCrystal` and `EntangledPairSource` parts) is the pair rate in pairs/s per
mW of pump power **emitted into the collection modes, before any downstream loss** (filters, fibre
coupling, detector efficiency). It is not a detected coincidence rate: entering a measured
coincidence rate there counts the losses twice. When given it replaces the computed rate of
`SPDCCrystal`; the computed rate carries the uncertainty described in
[docs/M3_REPORT.md](docs/M3_REPORT.md), section 6.

Things to know before trusting a number: [docs/conventions.md](docs/conventions.md) (quantum-layer
approximations) and [docs/M3_REPORT.md](docs/M3_REPORT.md) (physics model, validation, limitations).

## Test

```bash
uv run pytest
```
