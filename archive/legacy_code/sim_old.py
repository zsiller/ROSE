import json
import os
import pickle
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

import numpy as np

from helpers.log import get_logger
from workflows.run_context import RunContext
from simulations.sim_interface import simulate

from simulations.CDR_1D.sim_wrapper import run_cdr

logger = get_logger(__name__)

N_INITIAL = 1
WRITE_EVERY = 1000

def sim(X_path=None, Y_path=None, samples=None, sample_history=None):
    """Run CDR simulations and accumulate training data on disk.

    First call (betas=None): generates random initial betas.
    Subsequent calls: runs only the provided betas and appends.
    """
    if samples is None:
        logger.error("Samples must be provided")
        raise ValueError("Samples must be provided")

    if os.path.exists(samples):
        with open(samples, "rb") as f:
            samples = pickle.load(f)
    else:
        logger.error("Samples file not found: %s", samples)
        raise FileNotFoundError(f"Samples file not found: {samples}")
    if samples.size == 0:
        logger.error("Samples file is empty: %s", samples)
        raise ValueError("Samples file is empty")

    # Log the contents of the samples PKL file before running
    
    logger.info("Running %d simulation sample(s): %s", samples.size, samples.tolist())

    X_rows, Y_rows = [], []

    for s in samples:
        data = simulate("cdr",s)

    X_new = np.asarray(X_rows, dtype=float)
    Y_new = np.concatenate(Y_rows, axis=0)

    # Instead of overwriting, append to pkl files
    def append_to_pickle(filepath, new_data):
        # If the file exists and is not empty, load previous contents
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                # Try to stack if possible, otherwise just append the list
                if isinstance(data, np.ndarray) and isinstance(new_data, np.ndarray):
                    out = np.vstack([data, new_data])
                elif isinstance(data, list):
                    out = data + list(new_data)
                else:
                    # fallback: treat as list
                    out = [data, new_data]
            except Exception:
                # fallback, just use new_data if error
                out = new_data
        else:
            out = new_data
        with open(filepath, "wb") as f:
            pickle.dump(out, f)
        return out

    X = append_to_pickle(X_path, X_new)
    Y = append_to_pickle(Y_path, Y_new)

    if os.path.exists(sim_data.beta_history_path):
        with open(sim_data.beta_history_path, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.extend(betas.tolist())
    with open(sim_data.beta_history_path, "w") as f:
        json.dump(history, f)

    return {
        "Y_path": sim_data.Y_path,
        "X_path": sim_data.X_path,
        "beta_sample": betas.tolist(),
        "n_samples": X.shape[0],
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--x-path", required=True)
    parser.add_argument("--y-path", required=True)
    parser.add_argument("--new-sample-path", required=True)
    parser.add_argument("--sample-history-path", required=True)
    args = parser.parse_args()

    sim(X_path=args.x_path, Y_path=args.y_path, samples=args.new_sample_path, sample_history=args.sample_history_path)
