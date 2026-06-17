import os
import pickle
import sys
import argparse
import numpy as np
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

from helpers.log import get_logger
from helpers.resources import record_resources
from workflows.run_context import SubRunContext

from task_simulations.CDR_1D.sim_wrapper import run_cdr


script_map = {
        "cdr": "task_simulations/CDR_1D/sim_wrapper.py",
        "shock_tube_exact": "task_simulations/Shock_Tube/sod_exact.py",
        "shock_tube_euler": "task_simulations/Shock_Tube/sod_euler.py"
    }


def dispatch_problem(problem: str, data_path: str, params: list[float]) -> None:
    """
    Dispatch the problem by running the associated Python script via subprocess, passing params as CLI arguments.
    """
    try:
        script_path = script_map[problem]
    except KeyError as exc:
        raise ValueError(f"Unknown problem {problem!r}; known: {list(script_map.keys())}") from exc

    # Compose the command: python <script> <all-param-values> --output <output_path>
    cmd = [
        sys.executable,
        script_path,
        *map(str, params),
        "--h5", data_path,
        "--snapshot", "60"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Simulation command failed ({cmd}):\n{result.stderr}"
        )
    # Optionally parse/return output, or return None for file generation tasks
    return None


def main(data_path, samples, sample_history, model_name, run_tag="wf_0"):
    """Run CDR simulations and accumulate training data on disk.

    First call (betas=None): generates random initial betas.
    Subsequent calls: runs only the provided betas and appends.
    """
    logger = get_logger(__name__)
    if os.path.exists(samples):
        with open(samples, "rb") as f:
            samples = pickle.load(f)
    else:
        logger.error("Samples file not found: %s", samples)
        raise FileNotFoundError(f"Samples file not found: {samples}")

    if samples.size == 0:
        logger.error("Samples file is empty: %s", samples)
        raise ValueError("Samples file is empty")

    for s in samples:
        logger.info("Running simulation for sample: %s", s)
        data_path = os.path.join(data_path, run_tag)
        dispatch_problem(model_name, data_path, s)

    logger.info("Simulation completed for context: %s", data_path)

    # Add the new samples to the history pkl file
    if os.path.exists(sample_history):
        logger.info("Appending to existing sample history file: %s", sample_history)
        with open(sample_history, "ab") as f:
            pickle.dump(np.array(samples), f)
    else:
        logger.info("Creating new sample history file: %s", sample_history)
        with open(sample_history, "wb") as f:
            pickle.dump(np.array(samples), f)

    logger.info("Sample history completed for context: %s", sample_history)

    return {
        "data_path": data_path,
        "sample_history_path": sample_history,
        "n_samples": samples.shape
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", required=True, help="Path to context.json")
    args = parser.parse_args()

    ctx = SubRunContext.load(args.ctx)
    g, c, a = ctx.global_run_context, ctx.run_config, ctx.run_artifacts
    logger = get_logger(__name__)

    with record_resources(logger, "simulation"):
        main(
            data_path=g.data_dir,
            samples=a.new_sample_path,
            sample_history=a.sample_history_path,
            model_name=g.model_name,
            run_tag=c.wf_ID,
        )
