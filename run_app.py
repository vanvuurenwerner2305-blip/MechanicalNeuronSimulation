"""Launch the Membrane Neuron Simulator:  python run_app.py [model.step | project.mns | design.mad | neuron.mfn]
                     python run_app.py --analyse neuron.mfn   (opens it in the Analysis tab)"""
import sys

from app.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
