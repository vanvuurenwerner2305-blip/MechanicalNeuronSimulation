"""
Headless API of the Membrane Neuron Simulator, used through the `mns` command line (mns_api.cli) by people,
scripts and the design-study agent.

Everything works on a *study* folder (see mns_api.study): designs live in designs/<ID>_<name>/ with their
design file (design.yaml: parameters, CAD, roles), the generated CAD (model.step), the simulator project
(model.mns / model.mad / model.mfn, which the GUI opens), results/, renders/ and report/. Every command
appends events to .mns/events.jsonl, which the GUI's Study tab follows live.
"""
import os

os.environ.setdefault("QT_API", "pyqt5")
