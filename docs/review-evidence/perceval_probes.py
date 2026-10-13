"""Run with the isolated Perceval environment; no GUI input automation."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[3]
GUI = ROOT / "percevalGUI-v0 0 11"
sys.path.insert(0, str(GUI))
from circuit_builder import build_experiment, experiment_to_svg
from circuit_optimizer import merge_loss_channels, count_loss_channels
from circuit_file import load_setup
from components import FIELDS, parse_box
from sampling import make_input_state, run, keep_heralded

out = {}
values = dict(mode1=0, mode2=1, length1=0, length2=0, active_db=0)
noise = dict(brightness=1, g2=0, indistinguishability=1)
e = build_experiment([0,0], [("Beam Splitter Rx",values)],0,noise)
merged = merge_loss_channels(e)
for kind, texts in [("FockState",["1","1"]), ("NoisyFockState",["{0}","{1}"])]:
    state = make_input_state(kind,texts)
    result, perf = run(merged,"SLOS",state,"probs",None)
    out[kind] = {"results":result, "global_perf":perf,
                 "herald_mode0_equal2":keep_heralded(result,[(0,"=",2)])}
for detector in ["Threshold","PNR","PPNR","BSLayeredPPNR"]:
    d = dict(mode=0,length=0,efficiency=1,wires=4,layers=2,reflectivity=0.5)
    ed = build_experiment([0,0],[("Beam Splitter Rx",values),(detector,d)],0,noise)
    result, perf = run(merge_loss_channels(ed),"SLOS",make_input_state("FockState",["1","1"]),"probs",None)
    out[detector] = {"results":result, "global_perf":perf}
for path in sorted(GUI.glob("*.json")):
    raw = load_setup(path)
    m = len(raw["mode_losses"])
    parts = []
    for c in raw["components"]:
        fields = {key:parse_box(c["boxes"][key],kind,m) for key,label,kind in FIELDS[c["name"]]}
        parts.append((c["name"],fields))
    exp = build_experiment([float(x) for x in raw["mode_losses"]],parts,float(raw["db_per_cm"]),{k:float(v) for k,v in raw["noise"].items()})
    merged = merge_loss_channels(exp)
    out[path.name] = {"modes":m,"components":len(parts),"loss_channels_before":count_loss_channels(exp),
                      "loss_channels_after":count_loss_channels(merged),"svg_chars":len(experiment_to_svg(merged))}
    if path.name == "4p5m.json":
        result, perf = run(merged,"SLOS",make_input_state("FockState",["1","1","1","1","0"]),"probs",None)
        out[path.name].update(output_states=len(result),sum_probability=sum(result.values()),global_perf=perf)
print(json.dumps(out,indent=2))
