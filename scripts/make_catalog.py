"""Generate the typical-class part catalogue in library/ (idempotent).

Every value is a 'typical class' model (Sellmeier chromatic focal shift, super-Gaussian band
pass, coating bands, ...), tagged source=model; none is a specific vendor part.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sequoia.components.builtin  # noqa: E402,F401
from sequoia.library.editing import build_part  # noqa: E402
from sequoia.library.store import Library  # noqa: E402

LIB = Library(ROOT / "library")
MODEL = {"source": "model", "note": "typical class value, not a specific catalogue item"}
CREATED: list[str] = []


def curve(x, y, y_label="transmission", y_unit="1"):
    return {
        "x": [float(v) for v in x],
        "y": [float(v) for v in y],
        "x_label": "wavelength",
        "x_unit": "nm",
        "y_label": y_label,
        "y_unit": y_unit,
        "provenance": MODEL,
    }


def q(value, unit="1"):
    return {"value": float(value), "unit": unit, "provenance": MODEL}


def add(ref, type_name, model, params, description=None):
    if LIB.exists(ref):
        return
    header = {"manufacturer": "generic", "model": model, "description": description}
    LIB.save_part(build_part(ref, type_name, header, params))
    CREATED.append(ref)


def grid(lo=300.0, hi=1700.0, centers=(), fine=0.5, coarse=5.0, span=15.0):
    pts = set(np.arange(lo, hi + 1e-9, coarse))
    for c, width in centers:
        half = max(span * width, 5.0)
        pts |= set(np.arange(c - half, c + half + 1e-9, max(fine * width / 10, 0.05)))
    return np.array(sorted(pts))


def bandpass(c, fwhm, peak=0.9, od=5.0, order=3):
    x = grid(centers=[(c, fwhm)])
    t = peak * np.exp(-np.log(2.0) * (2.0 * np.abs(x - c) / fwhm) ** (2 * order))
    return x, np.maximum(t, 10.0**-od)


def edge(c, kind, peak=0.93, od=5.0, width=None):
    width = width or max(c * 0.01, 4.0)
    x = grid(centers=[(c, width)])
    s = 1.0 / (1.0 + np.exp(-(x - c) / (width / 4.0)))
    t = peak * (s if kind == "long" else 1.0 - s)
    return x, np.maximum(t, 10.0**-od)


def notch(c, width=20.0, od=6.0, peak=0.95):
    x = grid(centers=[(c, width)])
    t = peak - (peak - 10.0**-od) * np.exp(-np.log(2.0) * (2.0 * (x - c) / width) ** 8)
    return x, t


def sellmeier(kind):
    coef = {
        "nbk7": (
            (1.03961212, 0.231792344, 1.01046945),
            (0.00600069867, 0.0200179144, 103.560653),
        ),
        "fs": (
            (0.6961663, 0.4079426, 0.8974794),
            (0.0684043**2, 0.1162414**2, 9.896161**2),
        ),
    }[kind]

    def n(nm):
        l2 = (np.asarray(nm) / 1000.0) ** 2
        b, c = coef
        return np.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c, strict=True)))

    return n


def chromatic_focal(f_mm, glass="nbk7", reference=587.6):
    n = sellmeier(glass)
    x = np.arange(350.0, 1650.0 + 1, 10.0)
    f = f_mm * (n(reference) - 1.0) / (n(x) - 1.0)
    return curve(x, f * 1e-3, "focal length", "m")


def filters():
    for c, w in (
        (405, 10),
        (532, 10),
        (633, 1),
        (780, 10),
        (810, 1),
        (810, 3),
        (850, 40),
        (1064, 10),
    ):
        x, t = bandpass(c, w)
        add(
            f"filters/bp_{c}_{w}nm",
            "SpectralFilter",
            f"{c} nm band-pass, {w} nm FWHM, OD 5 blocking",
            {
                "transmission": curve(x, t),
                "damage_threshold_cw": 1000.0 if w >= 10 else 500.0,
            },
            "Typical-class thin-film band-pass; shifts to shorter wavelengths with angle.",
        )
    for c in (450, 550, 750, 850, 1000):
        x, t = edge(c, "long")
        add(
            f"filters/lp_{c}nm",
            "SpectralFilter",
            f"{c} nm long-pass edge filter, OD 5",
            {"transmission": curve(x, t), "damage_threshold_cw": 1000.0},
        )
    for c in (650, 750, 900):
        x, t = edge(c, "short", od=4.0)
        add(
            f"filters/sp_{c}nm",
            "SpectralFilter",
            f"{c} nm short-pass edge filter, OD 4",
            {"transmission": curve(x, t)},
        )
    for c in (532, 785):
        x, t = notch(c)
        add(
            f"filters/notch_{c}nm",
            "SpectralFilter",
            f"{c} nm notch filter, OD 6, 20 nm stop band",
            {"transmission": curve(x, t)},
        )
    for od in (0.3, 0.5, 2.0, 3.0, 4.0):
        name = str(od).replace(".", "p")
        add(
            f"filters/nd_filter_od{name}",
            "NDFilter",
            f"neutral density filter, OD {od:g}",
            {"optical_density": od, "damage_threshold_cw": 100.0},
        )


def lenses():
    for f in (25, 75, 150, 300, 500):
        add(
            f"freespace/lens_pcx_f{f}mm",
            "Lens",
            f"plano-convex lens f = {f} mm, N-BK7, 25.4 mm, AR 400-700 nm",
            {
                "focal_length": q(f, "mm"),
                "focal_length_curve": chromatic_focal(f),
                "clear_aperture": q(22.9, "mm"),
                "transmission": 0.99,
                "wavefront_error": 0.07,
                "damage_threshold_cw": 5000.0,
            },
            "Single N-BK7 element: the focal length changes with wavelength (chromatic shift).",
        )
    for f in (50, 100, 150):
        n = sellmeier("nbk7")
        x = np.arange(400.0, 1100.0 + 1, 10.0)
        shift = 1.0 + 0.0004 * ((n(x) - n(587.6)) / 0.01) ** 2
        add(
            f"freespace/lens_achromat_f{f}mm",
            "Lens",
            f"achromatic doublet f = {f} mm, 25.4 mm, AR 400-700 nm",
            {
                "focal_length": q(f, "mm"),
                "focal_length_curve": curve(x, f * 1e-3 * shift, "focal length", "m"),
                "clear_aperture": q(22.9, "mm"),
                "transmission": 0.99,
                "wavefront_error": 0.04,
                "damage_threshold_cw": 5000.0,
            },
            "Achromatic doublet: small residual chromatic focal shift.",
        )
    for f in (-50, -100):
        add(
            f"freespace/lens_pcv_f{abs(f)}mm",
            "Lens",
            f"plano-concave lens f = {f} mm, 25.4 mm",
            {
                "focal_length": q(f, "mm"),
                "focal_length_curve": chromatic_focal(f),
                "clear_aperture": q(22.9, "mm"),
                "transmission": 0.99,
                "wavefront_error": 0.07,
            },
        )
    add(
        "freespace/lens_aspheric_f11mm",
        "Lens",
        "aspheric lens f = 11 mm, NA 0.25, AR 600-1050 nm",
        {
            "focal_length": q(11, "mm"),
            "clear_aperture": q(5.5, "mm"),
            "transmission": 0.995,
            "wavefront_error": 0.02,
            "damage_threshold_cw": 2000.0,
        },
        "Aspheric collimation lens: almost no spherical aberration, small aperture.",
    )
    for mag, na, f in ((4, 0.1, 45.0), (20, 0.4, 9.0), (40, 0.65, 4.5), (60, 0.85, 3.0)):
        add(
            f"freespace/objective_{mag}x_na{str(na).replace('.', 'p')}",
            "Objective",
            f"{mag}x objective, NA {na}",
            {
                "effective_focal_length": q(f, "mm"),
                "numerical_aperture": na,
                "back_aperture": q(2 * f * na, "mm"),
                "working_distance": q(max(1.0, 20 * na**-0.5), "mm"),
                "transmission": 0.9,
                "wavefront_error": 0.05,
            },
        )


def mirrors():
    x = np.arange(300.0, 1700.0 + 1, 5.0)
    silver = np.where(x < 400, 0.2, 0.975 + 0.015 * (x > 800))
    add(
        "freespace/mirror_silver",
        "Mirror",
        "protected silver mirror, 25.4 mm",
        {
            "reflectance": curve(x, silver, "reflectance"),
            "clear_aperture": q(22.9, "mm"),
            "wavefront_error": 0.05,
            "damage_threshold_cw": 1000.0,
        },
    )
    alu = 0.9 - 0.04 * np.exp(-(((x - 810.0) / 40.0) ** 2))
    add(
        "freespace/mirror_aluminum",
        "Mirror",
        "protected aluminum mirror, 25.4 mm",
        {
            "reflectance": curve(x, np.where(x < 350, 0.88, alu), "reflectance"),
            "clear_aperture": q(22.9, "mm"),
            "wavefront_error": 0.05,
            "damage_threshold_cw": 500.0,
        },
    )
    gold = np.where(x < 520, 0.4, 0.97)
    add(
        "freespace/mirror_gold",
        "Mirror",
        "protected gold mirror, 25.4 mm",
        {
            "reflectance": curve(x, gold, "reflectance"),
            "clear_aperture": q(22.9, "mm"),
            "wavefront_error": 0.05,
            "damage_threshold_cw": 500.0,
        },
    )
    for lo, hi in ((400, 750), (750, 1100)):
        r = np.where((x >= lo) & (x <= hi), 0.995, 0.5)
        add(
            f"freespace/mirror_dielectric_{lo}_{hi}nm",
            "Mirror",
            f"broadband dielectric mirror {lo}-{hi} nm, 25.4 mm",
            {
                "reflectance": curve(x, r, "reflectance"),
                "clear_aperture": q(22.9, "mm"),
                "wavefront_error": 0.03,
                "damage_threshold_cw": 5000.0,
                "damage_threshold_pulsed": 1.0,
            },
        )
    for r_pct, name in ((90, "90_10"), (10, "10_90")):
        add(
            f"freespace/beamsplitter_{name}",
            "BeamSplitter",
            f"plate beam splitter R:T = {r_pct}:{100 - r_pct}",
            {
                "reflectance": r_pct / 100.0,
                "absorption": 0.01,
                "wavefront_error": 0.1,
            },
        )
    add(
        "freespace/beamsplitter_cube_50_50_pol_dependent",
        "BeamSplitter",
        "non-polarizing cube beam splitter 50:50 with 5% polarization dependence",
        {"reflectance": 0.5, "reflectance_v": 0.55, "absorption": 0.01, "wavefront_error": 0.05},
        "Rs differs from Rp, as for real 'non-polarizing' cubes.",
    )


def polarization():
    for wl in (405, 532, 633, 780, 810, 1064):
        for waves, tag in ((0.5, "hwp"), (0.25, "qwp")):
            add(
                f"polarization/{tag}_{wl}_zero_order",
                "WavePlate",
                f"zero-order {'half' if waves == 0.5 else 'quarter'}-wave plate, {wl} nm",
                {
                    "retardance": waves,
                    "design_wavelength": q(wl, "nm"),
                    "retardance_error": 1.0 / 300.0,
                    "transmission": 0.995,
                    "wavefront_error": 0.05,
                    "damage_threshold_cw": 1000.0,
                },
                "Zero-order plate, tolerance lambda/300; retardance varies with wavelength.",
            )
    add(
        "polarization/hwp_633_multi_order",
        "WavePlate",
        "multi-order half-wave plate, 633 nm (order 10)",
        {
            "retardance": 0.5,
            "order": 10,
            "design_wavelength": q(633, "nm"),
            "retardance_error": 1.0 / 200.0,
            "transmission": 0.99,
        },
        "Cheap multi-order plate: very sensitive to wavelength and temperature.",
    )
    x = np.arange(400.0, 1100.0 + 1, 10.0)
    for waves, tag, name in ((0.5, "hwp", "half"), (0.25, "qwp", "quarter")):
        add(
            f"polarization/{tag}_achromatic_500_900nm",
            "WavePlate",
            f"achromatic {name}-wave plate, 500-900 nm",
            {
                "retardance": waves,
                "design_wavelength": q(700, "nm"),
                "retardance_curve": curve(
                    x, waves * (1.0 + 0.01 * ((x - 700.0) / 200.0) ** 2), "retardance", "1"
                ),
                "transmission": 0.99,
            },
        )
    for er, name, t in ((1e5, "glan_taylor", 0.95), (1e3, "film", 0.8), (1e2, "wire_grid", 0.75)):
        add(
            f"polarization/polarizer_{name}",
            "Polarizer",
            f"{name.replace('_', '-')} polarizer, extinction ratio {er:g}",
            {
                "extinction_ratio": er,
                "transmission": t,
                "wavefront_error": 0.1,
                "damage_threshold_cw": 500.0 if name != "glan_taylor" else 2000.0,
            },
        )
    add(
        "polarization/pbs_cube_broadband",
        "PBS",
        "broadband polarizing beam splitter cube, Tp 0.95 / Rs 0.995, finite extinction",
        {
            "transmittance_p": 0.95,
            "transmittance_s": 0.005,
            "reflectance_s": 0.995,
            "reflectance_p": 0.05,
            "wavefront_error": 0.1,
            "damage_threshold_cw": 500.0,
        },
        "Typical cube PBS: transmission port is clean (1000:1), reflection port about 20:1.",
    )


def fibers():
    smf = {
        405: (3.3, 360, 0.12),
        488: (3.4, 430, 0.12),
        633: (4.3, 570, 0.13),
        980: (6.6, 920, 0.12),
        1064: (6.8, 970, 0.12),
        1550: (10.4, 1260, 0.14),
    }
    atten = {405: 40.0, 488: 20.0, 633: 12.0, 980: 2.0, 1064: 1.5, 1550: 0.2}
    for wl, (mfd, cutoff, na) in smf.items():
        add(
            f"fiber/smf_{wl}",
            "OpticalFiber",
            f"{wl} nm single-mode fiber",
            {
                "kind": "smf",
                "mode_field_diameter": q(mfd, "um"),
                "cutoff_wavelength": q(cutoff, "nm"),
                "numerical_aperture": na,
                "attenuation_db_per_km": atten[wl],
            },
        )
    for wl in (633, 1064, 1550):
        mfd, cutoff, na = smf[wl]
        add(
            f"fiber/pmf_{wl}",
            "OpticalFiber",
            f"{wl} nm polarization-maintaining fiber",
            {
                "kind": "pm",
                "mode_field_diameter": q(mfd * 1.05, "um"),
                "cutoff_wavelength": q(cutoff, "nm"),
                "numerical_aperture": na,
                "attenuation_db_per_km": atten[wl] * 1.3,
                "beat_length": q(2.5 if wl < 1000 else 4.0, "mm"),
                "polarization_extinction_db": 25.0,
            },
            "PM fiber: the slow axis acts as a long retarder; crosstalk limited by PER.",
        )
    for core, na in ((50, 0.2), (105, 0.22), (200, 0.22), (400, 0.39)):
        add(
            f"fiber/mmf_{core}um_na{str(na).replace('.', 'p')}",
            "OpticalFiber",
            f"{core} um core multimode fiber, NA {na}",
            {
                "kind": "mmf",
                "core_diameter": q(core, "um"),
                "numerical_aperture": na,
                "attenuation_db_per_km": 6.0,
            },
        )
    for f, name in ((4.5, "f4p5mm"), (7.5, "f7p5mm"), (15.0, "f15mm"), (37.5, "f37p5mm")):
        add(
            f"fiber/coupler_{name}",
            "FiberCoupler",
            f"FC fiber collimator / coupler f = {f} mm",
            {
                "focal_length": q(f, "mm"),
                "clear_aperture": q(f * 0.9, "mm"),
                "transmission": 0.98,
                "wavefront_error": 0.05,
            },
        )
    for wl, name in ((633, "633"), (1550, "1550")):
        add(
            f"fiber/beamsplitter_50_50_{name}",
            "FiberBeamSplitter",
            f"2x2 single-mode fiber coupler 50:50, {wl} nm",
            {"split_ratio": 0.5, "transmission": 0.97},
        )
    add(
        "fiber/beamsplitter_90_10_810",
        "FiberBeamSplitter",
        "2x2 single-mode fiber coupler 90:10, 810 nm",
        {"split_ratio": 0.9, "transmission": 0.97},
    )


def detectors_and_sources():
    x = np.arange(800.0, 1700.0 + 1, 10.0)
    resp = np.interp(
        x, [800, 900, 1000, 1300, 1550, 1620, 1700], [0.1, 0.45, 0.65, 0.9, 1.0, 0.95, 0.0]
    )
    add(
        "detectors/photodiode_ingaas",
        "Photodiode",
        "InGaAs photodiode, 3 mm active diameter",
        {
            "responsivity": curve(x, resp, "responsivity", "A/W"),
            "active_diameter": q(3, "mm"),
            "dark_current": q(5, "nA"),
        },
    )
    sx = np.arange(1000.0, 1700.0 + 1, 10.0)
    add(
        "detectors/snspd_telecom",
        "SNSPD",
        "SNSPD, fiber coupled, optimized for 1550 nm",
        {
            "efficiency": curve(
                sx, np.interp(sx, [1000, 1300, 1550, 1700], [0.5, 0.85, 0.9, 0.7]), "efficiency"
            ),
            "dark_count_rate": q(50, "Hz"),
            "dead_time": q(30, "ns"),
            "jitter_fwhm": q(25, "ps"),
        },
    )
    add(
        "detectors/spad_ingaas",
        "SPAD",
        "InGaAs/InP SPAD, gated, 1550 nm",
        {
            "efficiency": curve(
                sx, np.interp(sx, [1000, 1300, 1550, 1650], [0.1, 0.2, 0.25, 0.05]), "efficiency"
            ),
            "dark_count_rate": q(2000, "Hz"),
            "dead_time": q(5, "us"),
            "jitter_fwhm": q(200, "ps"),
            "afterpulse_probability": 0.05,
            "afterpulse_tau": q(1, "us"),
        },
    )
    for wl, power, name, lw in (
        (532, 20, "laser_532_cw", "1 MHz"),
        (633, 5, "laser_633_hene", "1.5 GHz"),
        (1550, 10, "laser_1550_cw", "100 kHz"),
    ):
        add(
            f"sources/{name}",
            "Laser",
            f"{wl} nm CW laser, {power} mW",
            {
                "wavelength": q(wl, "nm"),
                "power": q(power, "mW"),
                "linewidth": {
                    "value": float(lw.split()[0]),
                    "unit": lw.split()[1],
                    "provenance": MODEL,
                },
                "waist_radius": q(0.4, "mm"),
                "polarization_extinction_ratio": 500.0,
            },
        )
    add(
        "sources/laser_800_modelocked",
        "Laser",
        "800 nm mode-locked Ti:sapphire, 80 MHz, 100 fs, 1 W",
        {
            "wavelength": q(800, "nm"),
            "power": q(1000, "mW"),
            "waist_radius": q(1.0, "mm"),
            "repetition_rate": q(80, "MHz"),
            "pulse_width": q(100, "fs"),
            "pulse_shape": "sech2",
            "polarization_extinction_ratio": 100.0,
        },
    )

    for n in (1, 2):
        add(
            f"sources/fock_{n}photon_810",
            "FockSource",
            f"idealised Fock source, exactly {n} photon{'s' * (n > 1)} per pulse, 810 nm, 80 MHz",
            {
                "wavelength": q(810, "nm"),
                "photons_per_pulse": n,
                "repetition_rate": q(80, "MHz"),
                "pulse_width": q(1, "ps"),
                "waist_radius": q(0.4, "mm"),
                "distinguishability_tag": 0,
            },
            "Idealised model part: deterministic photon number, no loss, no multi-photon noise.",
        )


def phase_shifters():
    add(
        "polarization/phase_shifter_ideal",
        "PhaseShifter",
        "ideal optical phase shifter (lossless)",
        {},
        "Idealised model part: explicit phase on both polarizations or on one linear component.",
    )


def main() -> None:
    filters()
    lenses()
    mirrors()
    polarization()
    phase_shifters()
    fibers()
    detectors_and_sources()
    print(f"created {len(CREATED)} parts")
    for ref in CREATED:
        try:
            LIB.load_part(ref)
        except Exception as exc:
            print("INVALID", ref, exc)


if __name__ == "__main__":
    main()
