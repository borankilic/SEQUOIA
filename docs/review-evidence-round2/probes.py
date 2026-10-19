"""Independent round-two probes. Run from repository root with QT_QPA_PLATFORM=offscreen."""
import json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sequoia.fock import Circuit
from sequoia.fock.experiment import Experiment, run_experiment
from sequoia.fock.inputs import InputMixture, PureState
from sequoia.fock.analysis import outcome_table, condition
from sequoia.engines.montecarlo import TimeTags
from sequoia.analysis.correlation import analyse_tags, write_tags, read_tag_record
from sequoia.analysis.chsh import run_chsh, format_chsh
from sequoia.setup.io import load_setup
from sequoia.library.store import Library
out={}
mix=InputMixture(((0.25,PureState.fock([(0,0)])),))
r=run_experiment(Experiment(Circuit(1),[0],input_mixture=mix,shots=100,seed=1,rate_hz=100,duration_s=1))
out['truncated_sampling']={'mass':r.mass,'omitted':r.raw.omitted_weight,'probabilities':str(r.probabilities),'samples':str(r.samples.counts),'expected_counts':str(r.counts.total)}
r=run_experiment(Experiment(Circuit(1),[1],rate_hz=100,duration_s=0))
out['zero_duration']={'requested':0,'actual':r.counts.duration_s,'counts':str(r.counts.total)}
r=run_experiment(Experiment(Circuit(1),[1],rate_hz=100,duration_s=1))
out['uncertainty']={'expected_counts_sigma':r.counts.sigma[(1,)],'outcome_table_sigma':outcome_table(r.raw,n_windows=100)[0]['sigma']}
tags={'a':np.array([1e-6,2e-6]),'b':np.array([1e-6,2e-6])}
tt=TimeTags(tags,0.01,repetition_rates=[50e6,80e6],channel_clocks={})
sp=[{'type':'coincidences','name':'ab','channels':['a','b'],'window':'1 ns','center':'0 ns'}]
out['empty_channel_clocks']={'clock':tt.clock,'accidental_model':analyse_tags(tags,.01,sp,tt.clock)['ab']['accidental_model']}
p=ROOT/'docs/review-evidence-round2/mixed-clock.npz'
write_tags(p,tags,.01,tt.clock)
out['empty_channel_clocks']['reloaded_clock']=read_tag_record(p).repetition_rates
setup=load_setup(ROOT/'setups/bell_test.yaml');lib=Library(ROOT/'library')
for duration in [0.,-1.,10.]:
 c=run_chsh(setup,lib,'settings:hwp_a.angle','settings:hwp_b.angle',[0,22.5],[11.25,33.75],integration_time=duration)
 out[f'chsh_duration_{duration}']={'s':c.s,'sigma':str(c.s_sigma),'violates':c.violates,'text':format_chsh(c)}
from PySide6.QtWidgets import QApplication
from sequoia.gui.gaussian_panel import GaussianTab,GaussianOutcome
from sequoia.engines.gaussian import GaussianResult
from sequoia.gui.fock_panel import FockTab
app=QApplication.instance() or QApplication([])
g=GaussianTab(lambda:(None,None));g.set_outcome(SimpleNamespace(revision=1));g.begin_run();g.set_outcome(SimpleNamespace(revision=2))
g._done(GaussianOutcome(GaussianResult([],1.,{}, {}, {})))
out['edit_during_run']={'run_revision':g._run_revision,'latest_revision':g._latest_revision,'accepted':g.outcome is not None,'status':g.status.text()}
f=FockTab();f.set_experiment(Experiment(Circuit(1),[1],conditions={'keep':{0:1}},name='save me'))
out['editor_conditions_roundtrip']={'original':{'keep':{0:1}},'collected':f.experiment().conditions}
f.result=r;f._refresh_results();before=f.results_table.rowCount();f.reset();f._refresh_results()
out['editor_reset']={'rows_before':before,'result_survives':f.result is not None,'rows_after_refresh':f.results_table.rowCount()}
g.runner.shutdown();f.shutdown()
print(json.dumps(out,indent=2,default=str,allow_nan=False))
