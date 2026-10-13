from pathlib import Path
import json
import numpy as np
from sequoia.analysis.correlation import analyse_tags, write_tags, read_tags, coincidences

rng = np.random.default_rng(42)
rep, duration = 1e7, .01
pulses = np.arange(int(rep * duration)) / rep
tags = {name:pulses[rng.random(len(pulses)) < .1] for name in ['a','b']}
spec = [{'name':'ab','type':'coincidences','channels':['a','b'],'window':'1 ns','center':'0 ns'}]
live = analyse_tags(tags,duration,spec,[rep])['ab']
path = Path('/private/tmp/sequoia-review-pulsed-tags.npz')
write_tags(path,tags,duration)
loaded,span = read_tags(path)
reloaded = analyse_tags(loaded,span,spec)['ab']
delayed = {'h':np.arange(100)*1e-6+1e-6}
delayed['a'] = delayed['h']+100e-9
delayed['b'] = delayed['h']+101e-9
hbt = analyse_tags(delayed,1e-3,[{'type':'heralded_g2','name':'hbt','herald':'h','a':'a','b':'b','window':'2 ns'}])['hbt']
out = {
    'pulsed_tags_before_export':live,
    'same_tags_after_standard_reload':reloaded,
    'delayed_herald_100ns':hbt,
    'empty_tags_car':coincidences(np.array([]),np.array([]),1.0,1e-9,0.0),
}
print(json.dumps(out,indent=2))
