import html
from dataclasses import dataclass

ALL_TOOLS = "All tools (advanced)"


@dataclass(frozen=True)
class Recipe:
    name: str
    tools: tuple[str, ...]
    engine: str
    text: str
    assumptions: str
    polarization_view: int | None = None


RECIPES: dict[str, Recipe] = {
    r.name: r
    for r in (
        Recipe(
            "Build check",
            ("Diagnostics", "Links", "Readouts && probes"),
            "classical propagation and the rate engine, rerun after every edit",
            "Check that the drawn setup is complete before trusting any number: fix the errors in "
            "Diagnostics (click a row to select the part or link), follow power and beam size "
            "along Links, and read what detectors and probes see. Add a probe from the parts "
            "list onto a link to measure there.",
            "paraxial Gaussian beams, Jones polarization, mean powers and rates",
        ),
        Recipe(
            "Beam & alignment",
            ("Caustic", "Budget", "Spectra", "Polarization"),
            "classical propagation",
            "Pick a link, then check where the beam focuses and whether it fits the next part "
            "(Caustic), where light is lost and which part costs most (Budget), the spectrum and "
            "the polarization state. Sweeps (Compare & export) scan a lens position or plate "
            "angle against a coupling or power metric.",
            "paraxial Gaussian beams with M^2, no diffraction from clipping, mean powers",
        ),
        Recipe(
            "Source characterisation",
            ("Rate counts", "Histograms", "Gaussian", "Monte Carlo"),
            "rate engine (Rate counts); Gaussian and Monte Carlo for multi-pair and timing checks",
            "Commission a pair source: read singles, coincidences, CAR, heralding efficiency and "
            "heralded g2(0) in Rate counts, the coincidence peak in Histograms, multi-pair "
            "contamination versus pump power in Gaussian, and a simulated time-tag record in "
            "Monte Carlo to check the analytic numbers.",
            "Rate counts are first order in the pair rate with non-paralyzable dead time; "
            "Gaussian is per window without dead time; Monte Carlo includes jitter, dead time "
            "and noise",
        ),
        Recipe(
            "Two-photon interference",
            ("Rate counts", "Photon counts", "Sweeps"),
            "rate engine (delay-dependent HOM visibility); Photon counts for circuit-level "
            "photon-number interference",
            "Diagnose a HOM dip: the two-photon interference visibility (1 - C/C_distinguishable) "
            "is in Rate counts; sweep a delay or wave-plate angle in Sweeps to trace the dip. "
            "Photon counts gives output-state probabilities of the circuit, but without timing "
            "or spectral shape, so it cannot replace a delay scan.",
            "spectral-intensity overlap without spectral phase; Photon counts has no dead time",
        ),
        Recipe(
            "Polarization correlations / CHSH",
            ("CHSH", "Polarization"),
            "rate engine at the four analyser-setting pairs",
            "Characterise polarization entanglement: CHSH lists the four setting pairs (a b, "
            "a b', a' b, a' b') with the four coincidence analyses of each, gives E and S raw and "
            "accidental-subtracted, and exports the acquisition plan. The Polarization joint "
            "view shows the two-photon density matrix; a single photon of a Bell pair is "
            "unpolarized, so the single-link view does not test entanglement.",
            "coincidences post-selected on two detections (fair sampling); a forecast, not a "
            "measured violation",
            polarization_view=1,
        ),
        Recipe(
            "Photon-number / heralding",
            ("Photon counts",),
            "Fock backend (native or Perceval) through the setup's detectors",
            "Plan a photon-number experiment: Import from setup, Run, then read output-state "
            "probabilities, detector click patterns and heralded (conditional) distributions "
            "with their success probability and accepted-event rate.",
            "independent windows, no dead time (rates are potential clicks before dead time), "
            "truncated inputs with the omitted weight reported",
        ),
        Recipe(
            "Compare & export",
            ("Compare", "Sweeps"),
            "fits of measured data; sweeps of the drawn setup",
            "Load a measured CSV or XLSX, map its columns and units, fit a fringe, peak or HOM "
            "dip or compare with a sweep of the setup, then export data, model, residuals and a "
            "JSON record. Sweeps export their curves.",
            "fitted parameters are reported as not identifiable when the data cannot separate them",
        ),
    )
}

TASKS = (*RECIPES, ALL_TOOLS)


@dataclass(frozen=True)
class StartChoice:
    label: str
    detail: str
    template: str | None
    task: str


START_CHOICES = (
    StartChoice(
        "Align a beam into a fiber",
        "laser and fiber coupler; beam size, focus and coupling budget",
        "laser_fiber_coupling",
        "Beam & alignment",
    ),
    StartChoice(
        "Characterise a heralded source",
        "SPDC source with HBT detection; CAR, heralding efficiency, heralded g2(0)",
        "hbt_heralded",
        "Source characterisation",
    ),
    StartChoice(
        "Diagnose a HOM dip",
        "two photons on a beam splitter; visibility versus delay",
        "hom",
        "Two-photon interference",
    ),
    StartChoice(
        "Measure polarization correlations",
        "polarization-entangled pairs; four setting pairs, E and CHSH S",
        "bell_test",
        "Polarization correlations / CHSH",
    ),
    StartChoice(
        "Plan a photon-number experiment",
        "Fock-state inputs; click patterns and heralded distributions",
        "fock_hom",
        "Photon-number / heralding",
    ),
    StartChoice(
        "Compare with measured data",
        "a polarizer fringe to fit or compare with a sweep",
        "malus_law",
        "Compare & export",
    ),
    StartChoice(
        "Build a setup from scratch",
        "empty canvas; diagnostics, links and readouts",
        None,
        "Build check",
    ),
)


def recipe_text(name: str) -> str:
    recipe = RECIPES.get(name)
    if recipe is None:
        return ""
    e = html.escape
    return (
        f"<b>{e(recipe.name)}.</b> {e(recipe.text)}<br><b>Default engine:</b> "
        f"{e(recipe.engine)}. <b>Assumptions:</b> {e(recipe.assumptions)}."
    )
