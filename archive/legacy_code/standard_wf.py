"""Sequential workflow without ROSE — same executables as workflow.py for fair comparison."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from workflows.schema import clean, save

from helpers.timing import Timer

# Match workflow.py
AL_CYCLES = 11
MSE_THRESHOLD = 0.0000000001

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_script(rel_path: str) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / rel_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _run_script_stdout(rel_path: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / rel_path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def standard_wf() -> None:
    """Same step order as ROSE ``SequentialActiveLearner``: sim → train → AL → stop metric."""
    for i in range(AL_CYCLES):
        print(f"Iteration {i}")
        _run_script("simulations/sim.py")
        _run_script("surrogate/train.py")
        _run_script("active_learning/active_learning.py")

        out = _run_script_stdout("stop_criterion/check_mse.py")
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        mse = float(lines[-1])
        print(f"MSE: {mse}")
        if mse < MSE_THRESHOLD:
            break


if __name__ == "__main__":
    clean()

    timer = Timer()
    timer.enter()
    #print("starting workflow")
    standard_wf()
    #print("workflow completed")
    timer.exit()

    print(f"workflow completed in {timer.elapsed()} seconds")

    #save(result_dir="standard_wf")
