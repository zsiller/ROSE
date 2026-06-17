import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

from task_simulations.Shock_Tube.run_shock_tube import run_shock_tube
from simulations.sim_interface import simulate
from sklearn.model_selection import ParameterGrid
import numpy as np
from active_learning.lhs import lhs

default_params = np.array([1.0e5, 1.0e4, 1.0, 0.125])
low_params = default_params * 0.75
high_params = default_params * 1.25

NUM_SAMPLES = 500

OUT_DIR = Path(__file__).resolve().parent / "validation_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

canidates = lhs(NUM_SAMPLES, 4, low_params, high_params)

for canidate in canidates:
    simulate("shock_tube", canidate, str(OUT_DIR))

    