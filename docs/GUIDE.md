# Reading the results panels

SEQUOIA answers two questions about a setup: what does the light look like at every point
(beam, power, polarization), and what will the detectors count. The bottom panel has one tab for
each view.

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

## The tabs

| Tab | What it shows | How to read it |
|---|---|---|
| **Diagnostics** | Problems found in the setup (pump leaking to a detector, a fiber below cutoff, extrapolated data). | Click a row to select the culprit. Zero rows is good. |
| **Links** | For every link: power, beam radius, polarization purity (DOP), photons per second. | Follow a beam down the table to see where light is lost. After the long-pass filter the pump power drops by orders of magnitude. |
| **Readouts and probes** | What detectors and probes read. | Photocurrent for a photodiode, power for a power meter. |
| **Counts** | Clicks per second per detector, then the coincidence analyses you defined. | CAR, heralding efficiency and window fractions are the figures of merit of a pair source. |
| **Caustic** | Beam radius along the path. | A dip is a focus. Check that the beam is not larger than the next part. |
| **Polarization** | Polarization of the light on a link, as an ellipse and as a point on the Poincare sphere (drag to rotate). | Linear states sit on the equator, circular at the poles, a point inside means partially polarized. |
| **Spectra** | Power versus wavelength on a link. | Filters show up as cut or narrowed spectra. |
| **Histograms** | Arrival-time difference between two detectors. | A peak at zero delay is correlated pairs; the flat floor is accidentals. Its height over the floor is the CAR. |
| **Budget** | Power or photon flux along a path, plus what the result would be with each part replaced by an ideal one. | Parts at the top of the list cost the most; improve those first. |
| **Sweeps** | Vary one setting (a wave-plate angle, a lens distance) and plot a result; the optimizer finds the best value. | A sweep of the half-wave plate in the Bell test shows the polarization fringe. |
| **Monte Carlo** | A simulated stream of individual detector clicks with jitter, dead time and noise. | It should match the analytic Counts within the shown error (z within 3). Use it when dead time or timing matters. |
| **Gaussian** | Exact photon statistics when several pairs can be created at once. | CAR falls and heralded g2 rises as you turn up the pump; this is why the pump power is kept low in these experiments. |

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

Splitters whose transmitted and reflected fractions add up to more than 1 are rejected.
`scripts/make_catalog.py` regenerates the typical-class parts (filters, lenses, mirrors, wave
plates, polarizers, fibers, detectors, lasers) with these properties.
