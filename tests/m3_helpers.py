import math
from pathlib import Path

import numpy as np
from scipy import constants

import sequoia.components.builtin  # noqa: F401
from sequoia.components import nonlinear
from sequoia.data import Scalar
from sequoia.engines.rates import simulate_rates
from sequoia.library.store import Library
from sequoia.setup.io import load_setup
from sequoia.setup.resolve import instantiate

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Library(ROOT / "library")
C = constants.c
H_PLANCK = constants.h


def load_example(name):
    return load_setup(ROOT / "setups" / f"{name}.yaml").model_copy(deep=True)


def instance(setup, inst_id):
    return next(i for i in setup.instances if i.id == inst_id)


def link(setup, link_id):
    return next(link for link in setup.links if link.id == link_id)


def set_settings(setup, inst_id, **settings):
    inst = instance(setup, inst_id)
    inst.settings = {**inst.settings, **settings}


def set_overrides(setup, inst_id, **overrides):
    inst = instance(setup, inst_id)
    inst.overrides = {**inst.overrides, **overrides}


def set_length(setup, link_id, meters):
    link(setup, link_id).length = Scalar(value=float(meters), unit="m")


def remove_instance(setup, inst_id, bridge=None):
    setup.instances = [i for i in setup.instances if i.id != inst_id]
    kept = []
    for lk in setup.links:
        if lk.src.split(".")[0] == inst_id or lk.dst.split(".")[0] == inst_id:
            continue
        kept.append(lk)
    setup.links = kept
    if bridge is not None:
        setup.links.append(bridge)


def part(ref):
    return LIBRARY.load_part(ref).params


class EmissionCache:
    """Memoises SPDCCrystal.emit_pairs so sweeps over downstream parameters do not
    re-evaluate the (slow) spdcalc joint spectrum. Keyed on the crystal parameters,
    settings and incident pump light."""

    def __init__(self, monkeypatch):
        self.original = nonlinear.SPDCCrystal.emit_pairs
        self.store = {}
        self.calls = 0
        cache = self

        def cached(self_, lights, ctx):
            light = lights.get("in")
            key = (
                self_.params.model_dump_json(),
                self_.settings.model_dump_json(),
                None
                if light is None
                else (
                    round(light.power, 15),
                    tuple(np.round(light.wavelengths * 1e12, 6)),
                    complex(light.mode.q_x),
                    complex(light.mode.q_y),
                    tuple(np.round(light.coherency.reshape(-1).real, 9)),
                ),
            )
            if key not in cache.store:
                cache.calls += 1
                local = type(ctx)(issues=[], subject=ctx.subject, instance_id=ctx.instance_id)
                emission = cache.original(self_, lights, local)
                cache.store[key] = (emission, local.issues)
            emission, issues = cache.store[key]
            ctx.issues.extend(issues)
            return emission

        monkeypatch.setattr(nonlinear.SPDCCrystal, "emit_pairs", cached)


def run(setup):
    return simulate_rates(setup, LIBRARY)


def crystal_emission(setup, result, crystal_id="crystal", link_id="s_focus"):
    inst = instance(setup, crystal_id)
    comp = instantiate(LIBRARY, inst)
    from sequoia.components.base import Context

    light = result.classical.links[link_id].end
    return comp.emit_pairs({"in": light}, Context(subject=crystal_id))


def q_chain(wavelength, w0, ops, z0=0.0):
    q = z0 + 1j * math.pi * w0**2 / wavelength
    for kind, value in ops:
        if kind == "d":
            q = q + value
        elif kind == "f":
            q = 1.0 / (1.0 / q - 1.0 / value)
        else:
            raise ValueError(kind)
    return q


def radius_and_curvature(q, wavelength):
    inv = 1.0 / q
    w = math.sqrt(-wavelength / (math.pi * inv.imag))
    r = math.inf if inv.real == 0.0 else 1.0 / inv.real
    return w, r


def gaussian_overlap(q1, q2, wavelength):
    w1, r1 = radius_and_curvature(q1, wavelength)
    w2, r2 = radius_and_curvature(q2, wavelength)
    curvature = (1.0 / r1 if math.isfinite(r1) else 0.0) - (1.0 / r2 if math.isfinite(r2) else 0.0)
    return 4.0 / ((w1 / w2 + w2 / w1) ** 2 + (math.pi * w1 * w2 / wavelength) ** 2 * curvature**2)


def marginals(source):
    """Signal and idler marginal spectra of a PairSource as (wavelength, weight) arrays."""
    w = np.abs(source.amplitude) ** 2
    sig = w.sum(axis=(0, 1, 3))
    lam_s = C / (source.n_signal * source.dnu)
    n_i = source.n_sum[None, :] - source.n_signal[:, None]
    grid, inverse = np.unique(n_i, return_inverse=True)
    idl = np.zeros(len(grid))
    np.add.at(idl, inverse.reshape(n_i.shape), w.sum(axis=(0, 1)))
    lam_i = C / (grid * source.dnu)
    return (lam_s, sig / sig.sum()), (lam_i, idl / idl.sum())


def band_average(spectrum, *curves):
    lam, weight = spectrum
    value = np.ones_like(lam)
    for curve in curves:
        value = value * np.asarray(curve(lam), dtype=float)
    return float(np.sum(weight * value))


def fiber_transmission(fiber_params, length_m, connector_db=0.0):
    db = fiber_params.attenuation_db_per_km * length_m / 1e3 + connector_db
    return 10.0 ** (-db / 10.0)


def window_fraction(window, fwhms):
    sigma = math.sqrt(sum(f**2 for f in fwhms)) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return math.erf(window / 2.0 / (math.sqrt(2.0) * sigma))


def photon_rate(power_w, wavelength):
    return power_w / (H_PLANCK * C / wavelength)
