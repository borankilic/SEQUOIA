# SEQUOIA

**S**imulated **E**valuation of **QU**antum **O**ptics **I**nstruments and **A**nalysis: a desktop simulator for quantum optics lab setups.

- Implementation plan: [docs/PLAN.md](docs/PLAN.md)
- Physics and code conventions: [docs/conventions.md](docs/conventions.md)

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
from sequoia.analysis import metrics          # car, chsh_s, tomography_settings, mle_density_matrix, ...
result = simulate_rates(setup, library)
result.analyses["herald_a"]["car"], result.channels["tagger.ch1"]["singles_hz"]
```

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
