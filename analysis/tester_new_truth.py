import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from task_simulations.Shock_Tube.sod_euler import EulerSolver1D

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Some surrogate.pkl files (e.g. run_200) were pickled when the surrogate package
# was named `surrogate`; the classes now live under `task_train`. Alias the old
# module names so pickle.load can resolve `surrogate.model` / `surrogate.POD`.
import task_train
import task_train.model
import task_train.POD

sys.modules.setdefault("surrogate", task_train)
sys.modules.setdefault("surrogate.model", task_train.model)
sys.modules.setdefault("surrogate.POD", task_train.POD)


import random

# Default values
DEFAULT_DATA_POINT = [100000, 10000, 1.0, 0.125, 0.0006]

NAME = "run_200"

SURROGATE_FILE = "/home/zhsiller/research/ROSE/training_runs/shock_tube/run_200/wf_0/surrogate.pkl"
SHOCK_TUBE_DIR = Path("/home/zhsiller/research/ROSE/task_simulations/Shock_Tube")
EXACT_DIR = SHOCK_TUBE_DIR / "sod_exact.py"
TRUTH_DIR = SHOCK_TUBE_DIR / "sod_euler.py"

def random_data_point_within_bounds(defaults, bounds=0.25):
    result = []
    for i, val in enumerate(defaults):
        if i < 2:  # first two params: int (or float, but use ints)
            lower = int(val * (1 - bounds))
            upper = int(val * (1 + bounds))
            result.append(random.randint(lower, upper))
        elif 2 <= i < 4:  # next two: floats, rounded to 3 decimals
            lower = val * (1 - bounds)
            upper = val * (1 + bounds)
            random_val = random.uniform(lower, upper)
            result.append(round(random_val, 3))
        else:
            # fifth param remains the same (or could add more logic if desired)
            result.append(val)
    return result

DATA_POINT = random_data_point_within_bounds(DEFAULT_DATA_POINT)
print(DATA_POINT)

# DATA_POINT = [115937, 8176, .885, .144, 0.0006]

def gen_rho(data_point, dir_name, prefix):
    script_name = f"{prefix}.py"
    subprocess.run(
        [
            sys.executable,
            str(dir_name / script_name),
            *map(str, data_point),
            "--out",
            f"{prefix}_density.npz",
        ],
        cwd=dir_name,
        check=True,
    )

    data = np.load(dir_name / f"{prefix}_density.npz")
    x_src = data["x"]
    rho_src = data["rho"]
    n_cells = 256
    x_cells = (np.arange(n_cells, dtype=float) + 0.5) / n_cells
    return np.interp(x_cells, x_src, rho_src.flatten())



if __name__ == "__main__":

    with open(SURROGATE_FILE, "rb") as f:
        surrogate = pickle.load(f)

    rho_pred, std = surrogate.predict(np.asarray([DATA_POINT], dtype=np.float64), return_std=True)
    rho_pred = rho_pred[0]

    exact = gen_rho(DATA_POINT, SHOCK_TUBE_DIR, "sod_exact")
    truth = gen_rho(DATA_POINT, SHOCK_TUBE_DIR, "sod_euler")

    plt.figure(figsize=(8, 6))
    plt.plot(rho_pred, label="surrogate", linestyle="-", linewidth=2)
    plt.plot(exact, label="analytical", linestyle="--", color="black", linewidth=2)
    plt.plot(truth, label="numerical", linestyle=":", color="red", linewidth=2)
    plt.legend()
    plt.xlabel("cell index")
    plt.ylabel("rho")
    plt.title("rho profiles")
    plt.suptitle(f"p=({DATA_POINT[0]:.0f}, {DATA_POINT[1]:.0f}), rho=({DATA_POINT[2]:.3f}, {DATA_POINT[3]:.3f})")
    plt.grid(True)
    plt.savefig(f"rho_profiles_{NAME}.png")

    # time evolution "color plot" (space vs. time vs. rho) with same params
    times = np.linspace(0, DATA_POINT[4], 64)
    nx = 256
    cell_indices = np.arange(nx)
    rhos_evolution = []
    error_exact_evolution = []
    error_truth_evolution = []

    for t in times:
        # For each time, generate the surrogate prediction for current t
        datapoint_t = list(DATA_POINT[:4]) + [t]
        rho_pred_t, _ = surrogate.predict(
            np.asarray([datapoint_t], dtype=np.float64), return_std=True
        )
        rho_pred_t = rho_pred_t[0]
        rhos_evolution.append(rho_pred_t.copy())

        # Error of the surrogate prediction against the exact and truth solutions at this t
        exact_t = gen_rho(datapoint_t, SHOCK_TUBE_DIR, "sod_exact")
        truth_t = gen_rho(datapoint_t, SHOCK_TUBE_DIR, "sod_euler")
        error_exact_evolution.append(rho_pred_t - exact_t)
        error_truth_evolution.append(rho_pred_t - truth_t)
    rhos_evolution = np.array(rhos_evolution)
    error_exact_evolution = np.array(error_exact_evolution)
    error_truth_evolution = np.array(error_truth_evolution)

    plt.figure(figsize=(10, 7))
    plt.imshow(
        rhos_evolution,
        aspect='auto',
        extent=[0, nx, 0, DATA_POINT[4]],   # x axis is cell index now
        origin='lower',
        cmap='viridis'
    )
    plt.colorbar(label='rho')
    plt.xlabel("cell index")
    plt.ylabel("time")
    plt.title(
        f"Time Evolution of Density (rho)\np=({DATA_POINT[0]:.0f}, {DATA_POINT[1]:.0f}), "
        f"rho=({DATA_POINT[2]:.3f}, {DATA_POINT[3]:.3f})"
    )
    plt.savefig(f"rho_time_evolution_{NAME}.png")

    # space-time error plots (prediction - exact) and (prediction - truth),
    # sharing one color scale so they are directly comparable
    err_max = max(
        np.abs(error_exact_evolution).max(),
        np.abs(error_truth_evolution).max(),
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    im = None
    for ax, err, ref in (
        (axes[0], error_exact_evolution, "analytical"),
        (axes[1], error_truth_evolution, "numerical"),
    ):
        im = ax.imshow(
            err,
            aspect='auto',
            extent=[0, nx, 0, DATA_POINT[4]],   # x axis is cell index
            origin='lower',
            cmap='coolwarm',
            vmin=-err_max,
            vmax=err_max,
        )
        ax.set_xlabel("cell index")
        ax.set_title(f"surrogate - {ref}")
    axes[0].set_ylabel("time")

    # single colorbar shared by both error panels
    cbar = fig.colorbar(im, ax=axes, label="rho error")
    # center the title over the two panels (not the whole figure, which the
    # colorbar shifts off-center)
    panel_center = 0.5 * (axes[0].get_position().x0 + axes[1].get_position().x1)
    fig.suptitle(
        f"Space-Time Error of Density (rho)\n"
        f"p=({DATA_POINT[0]:.0f}, {DATA_POINT[1]:.0f}), "
        f"rho=({DATA_POINT[2]:.3f}, {DATA_POINT[3]:.3f})",
        x=panel_center,
    )
    fig.savefig(f"rho_time_error_{NAME}.png", bbox_inches="tight")
