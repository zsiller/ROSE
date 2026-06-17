"""Run the workflow multiple times for uncertainty and random sampling,
collect RMSE per iteration, average across runs, and plot."""

import subprocess
import sys
import re
import json

import numpy as np
import matplotlib.pyplot as plt

N_RUNS = 5
AL_CYCLES = 11
METHODS = ["uncertainty", "random"]
AL_FILE = "active_learning/active_learning.py"


def set_method(method):
    """Swap the METHOD variable in active_learning.py."""
    path = AL_FILE
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r'^METHOD\s*=\s*".*"', f'METHOD = "{method}"', content, flags=re.MULTILINE)
    with open(path, "w") as f:
        f.write(content)


def run_workflow(debug=False):
    """Run workflow.py and parse RMSE values from stdout."""
    result = subprocess.run(
        [sys.executable, "workflow.py"],
        capture_output=True, text=True
    )
    if debug or result.returncode != 0:
        print(f"\n    [exit code: {result.returncode}]")
        if result.stderr:
            print(f"    [stderr]: {result.stderr[-1000:]}")
        if result.stdout:
            print(f"    [stdout]: {result.stdout[:500]}")
    rmse_values = []
    for line in result.stdout.splitlines():
        match = re.match(r"Iteration\s+(\d+):\s+([\d.e\-+]+)", line)
        if match:
            rmse_values.append(float(match.group(2)))
    return rmse_values


def main():
    results = {}

    for method in METHODS:
        print(f"\n{'='*50}")
        print(f"Running {N_RUNS} trials with method: {method}")
        print(f"{'='*50}")

        set_method(method)
        all_runs = []

        for run in range(N_RUNS):
            print(f"  Run {run + 1}/{N_RUNS}...", end=" ", flush=True)
            rmse = run_workflow(debug=(run == 0))
            print(f"final RMSE={rmse[-1]:.6f}" if rmse else "no data")
            # Pad to AL_CYCLES if stopped early
            while len(rmse) < AL_CYCLES:
                rmse.append(rmse[-1] if rmse else float("nan"))
            all_runs.append(rmse[:AL_CYCLES])

        results[method] = np.array(all_runs)

    # Compute mean and std
    iterations = np.arange(AL_CYCLES)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"uncertainty": "tab:blue", "random": "tab:orange"}
    labels = {"uncertainty": "Uncertainty Sampling", "random": "Random Sampling"}
    markers = {"uncertainty": "o", "random": "s"}

    for method in METHODS:
        data = results[method]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        ax.plot(iterations, mean, f"{markers[method]}-", color=colors[method],
                linewidth=2, label=labels[method])
        ax.fill_between(iterations, mean - std, mean + std,
                         color=colors[method], alpha=0.2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Active Learning Comparison ({N_RUNS} runs avg ± 1 std)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/mse_benchmark_2.png", dpi=200)
    plt.close()
    print(f"\nSaved figures/mse_benchmark_2.png")

    # Save raw data
    with open("benchmark_results_2.json", "w") as f:
        json.dump({m: results[m].tolist() for m in METHODS}, f)
    print("Saved benchmark_results_2.json")


if __name__ == "__main__":
    main()
