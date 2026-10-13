"""Reproducible review probes; run from SEQUOIA with venv/bin/python.

These print observations, not regression-test assertions or proposed fixes.
"""
from pathlib import Path
import sys
import math
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_gaussian_engine import batch_from
from sequoia.engines.gaussian import (
    GaussianModel, GaussianSource, GaussianSetup, _Member, _detector_forms, _pnr,
    prepare_gaussian,
)
from sequoia.library.store import Library
from sequoia.setup.io import setup_from_dict, load_setup
from sequoia.engines.rates import simulate_rates
from sequoia.engines.classical import simulate

LIB = Library(ROOT / "library")
observations = {}

# Exactly solvable single thermal marginal of a two-mode squeezed state.
j = np.zeros((2, 2), complex)
j[0, 0] = 1
hs = np.zeros((1, 1, 2, 2), complex)
hi = np.zeros_like(hs)
hs[0, 0] = np.eye(2)
batch, hs, hi = batch_from(j, hs, hi)
r = math.asinh(math.sqrt(10.0))
member = _Member(1.0, [batch])
source = GaussianSource("s", r*r, 1.0, False, [member], 1)
model = GaussianModel([source], {"s": {id(member): _detector_forms([batch], hs, hi)}}, ["d"], np.zeros(1))
probs = _pnr(model, ["d"], ("d",), 1, 12)
x = 10 / 11
observations["pnr_thermal_mean_10_cutoff_12"] = {
    "reported_p0": float(probs[0]), "exact_p0": 1/11,
    "reported_sum": float(probs.sum()), "exact_retained_mass": 1-x**13,
    "exact_tail": x**13,
    "reported_mean": float(np.dot(np.arange(13), probs)), "exact_mean": 10,
}

def simple_source(dnu="1 GHz", prefix="a", channel=1):
    s, d = prefix+"s", prefix+"d"
    return ([
        {"id": s, "part": "nonlinear/entangled_810_generic", "overrides": {"dnu":dnu}},
        {"id": d, "part":"detectors/snspd_generic", "settings":{"input":"free_space"},
         "overrides":{"efficiency":1, "dark_count_rate":"0 Hz", "dead_time":"0 ns"}},
    ], [
        {"id":prefix+"l", "kind":"free_space", "from":s+".signal", "to":d+".in", "length":"0 m"},
        {"id":prefix+"c", "kind":"cable", "from":d+".out", "to":f"t.ch{channel}", "delay":"0 ns"},
    ])

i1,l1 = simple_source()
i2,l2 = simple_source("2 GHz", "b", 2)
tagger = {"id":"t", "part":"electronics/tagger_generic_4ch"}
one = setup_from_dict({"instances":i1+[tagger], "links":l1})
two = setup_from_dict({"instances":i1+i2+[tagger], "links":l1+l2})
g1 = prepare_gaussian(one, LIB).evaluate()
g2 = prepare_gaussian(two, LIB).evaluate()
observations["independent_source_changes_existing_rate"] = {
    "single_source_window":g1.window, "two_source_window":g2.window,
    "old_channel_alone_Hz":g1.mean_photons["t.ch1"]/g1.window,
    "old_channel_with_independent_source_Hz":g2.mean_photons["t.ch1"]/g2.window,
    "issues":[i.code for i in g2.issues],
}

empty_dets = setup_from_dict({"instances":[i1[0]]})
try:
    prepare_gaussian(empty_dets, LIB)
    observations["pair_source_without_detectors"] = "accepted"
except Exception as e:
    observations["pair_source_without_detectors"] = f"{type(e).__name__}: {e}"

# A coherent-light-only detector tests whether the advertised pump factor scales leakage.
laser = setup_from_dict({"instances":[
    {"id":"laser", "part":"sources/laser_780_cw", "overrides":{"power":"1 fW"}},
    i1[1], tagger], "links":[
    {"id":"light", "kind":"free_space", "from":"laser.out", "to":"ad.in", "length":"0 m"}, l1[1]]})
prepared = prepare_gaussian(laser, LIB)
observations["gain_scale_background"] = {str(scale):prepared.evaluate(scale).mean_photons for scale in [0,1,10]}

zero = GaussianSetup(GaussianModel([], {}, ["a","b"], np.zeros(2)), ["a","b"], 1.0,
    [{"name":"ab", "type":"coincidences", "channels":["a","b"]}], [], {}).evaluate()
observations["zero_light_car"] = zero.analyses

# Record engine baselines for every shipped example; source-only polarization display uses classical state.
for name in ["bell_test", "hom", "hbt_heralded", "mzi", "laser_fiber_coupling"]:
    s = load_setup(ROOT / "setups" / f"{name}.yaml")
    c = simulate(s, LIB)
    rr = simulate_rates(s, LIB, classical=c)
    observations["baseline_"+name] = {
        "classical_links":len(c.links), "channels":len(rr.channels),
        "issues":[i.code for i in c.issues+rr.issues],
        "analyses": {n:{k:v for k,v in a.items() if isinstance(v,(str,int,float))} for n,a in rr.analyses.items()},
    }
    if name == "bell_test":
        observations["bell_polarization_link"] = {
            "classical_end_is_none": c.links["s_a1"].end is None,
            "pair_photons_Hz":rr.links["s_a1"].get("photons_hz"),
        }
print(json.dumps(observations, indent=2, default=str))
