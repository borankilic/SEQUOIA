import json
from sequoia.fock.detectors import IdealPNR, PseudoPNR, Threshold
from sequoia.fock.experiment import Experiment, DetectorSpec
from sequoia.fock.circuit import Circuit
from PySide6.QtWidgets import QApplication
from sequoia.gui.fock_panel import FockTab
out={}
for d in [IdealPNR(dark=.2),PseudoPNR(wires=4,dark=.2),Threshold(dark=.2)]:
 out[d.kind]={'vacuum_response':d.response_matrix(0).tolist(),'one_photon_response':d.response_matrix(1).tolist()}
app=QApplication([]);f=FockTab()
d=PseudoPNR(wires=2,weights=(.9,.1),wire_efficiencies=(.5,.8),dark=.2)
e=Experiment(Circuit(1),[1],detectors=[DetectorSpec((0,),d)],conditions={'herald':{0:1}},description='preserve this')
f.set_experiment(e);r=f.experiment()
out['editor_roundtrip']={'before_detector':d.to_dict(),'after_detector':r.detectors[0].model.to_dict(),'before_conditions':e.conditions,'after_conditions':r.conditions,'description':r.description}
print(json.dumps(out,indent=2));f.shutdown()
