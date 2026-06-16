"""Shared infrastructure for the Dakota MCMC shock-tube workflow.

Everything under ``dakota_mcmc/`` imports from here instead of ``MCMC/`` or
``inference/``. The only outside dependency is the physics stack in
``task_simulations/Shock_Tube/`` (and the surrogate pickle for
``SOD_FORWARD=surrogate``).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_simulations.Shock_Tube.sod_exact import (  # noqa: E402
    exact_density_on_cells,
    shock_features,
)

N_PARAMS = 4
N_FIELD = 256
FIELD_OFFSET = N_PARAMS + 1
N_STATE = FIELD_OFFSET + N_FIELD
T_FINAL = 6.0e-4

PARAM_NAMES = ["p_high", "p_low", "rho_high", "rho_low"]
TRUTH = np.array([1.0e5, 0.1e5, 1.0, 0.125])
OBS_ERROR = 0.01

ENSEMBLE_DIR = ROOT / "training_data" / "shock_tube" / "enkf_ensemble_files"

FORWARD_MODES = ("exact", "euler", "surrogate")
SURROGATE_PKL = ROOT / "training_runs" / "shock_tube" / "run_200" / "wf_0" / "surrogate.pkl"

_surrogate_cache: dict[Path, object] = {}
GAMMA = 1.4

_SOD_EXACT = ROOT / "task_simulations" / "Shock_Tube" / "sod_exact.py"
_SOD_EULER = ROOT / "task_simulations" / "Shock_Tube" / "sod_euler.py"
_SCRATCH_NPZ = Path(tempfile.gettempdir()) / f"dakota_forward_{os.getpid()}.npz"


def _forward_npz(script: Path, params, t: float, out_path: Path = _SCRATCH_NPZ) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    cmd = [sys.executable, str(script), *(f"{v:.10g}" for v in p), f"{t:.10g}",
           "--nx", str(N_FIELD), "--out", str(out_path), "--quiet", "--fast"]
    subprocess.run(cmd, check=True)
    with np.load(out_path) as data:
        return np.asarray(data["rho"], dtype=float)


def _euler_in_process(params, t: float) -> np.ndarray:
    from task_simulations.Shock_Tube.sod_euler import EulerSolver1D
    p = np.asarray(params, dtype=float)
    solver = EulerSolver1D(nx=N_FIELD, xmin=0.0, xmax=1.0, gamma=GAMMA, cfl=0.5)
    solver.set_sod_like(rho_high=p[2], p_high=p[0], rho_low=p[3], p_low=p[1], x0=0.5)
    solver.step_to(t)
    return solver.U[0]


def exact_density(params, t: float = T_FINAL, *, in_process: bool = False) -> np.ndarray:
    if in_process:
        return exact_density_on_cells(params, t, N_FIELD)
    return _forward_npz(_SOD_EXACT, params, t)


def euler_density(params, t: float = T_FINAL, *, in_process: bool = False) -> np.ndarray:
    if in_process:
        return _euler_in_process(params, t)
    return _forward_npz(_SOD_EULER, params, t)


def load_surrogate(path: Path | str | None = None):
    import pickle

    import task_train
    import task_train.model
    import task_train.POD

    sys.modules.setdefault("surrogate", task_train)
    sys.modules.setdefault("surrogate.model", task_train.model)
    sys.modules.setdefault("surrogate.POD", task_train.POD)

    path = Path(path) if path is not None else SURROGATE_PKL
    if path not in _surrogate_cache:
        with open(path, "rb") as fh:
            _surrogate_cache[path] = pickle.load(fh)
    return _surrogate_cache[path]


def surrogate_density(params, t: float = T_FINAL, *, pkl: Path | str | None = None) -> np.ndarray:
    sur = load_surrogate(pkl)
    X = np.concatenate([np.asarray(params, dtype=float), [float(t)]])[None, :]
    Y, _ = sur.predict(X)
    return np.asarray(Y, dtype=float)[0, :N_FIELD]


def make_forward(mode: str, cell_idx: np.ndarray | None = None, t: float = T_FINAL,
                 *, in_process: bool = False):
    if mode == "exact":
        full = lambda m: exact_density(m, t, in_process=in_process)
    elif mode == "euler":
        full = lambda m: euler_density(m, t, in_process=in_process)
    elif mode == "surrogate":
        load_surrogate()
        full = lambda m: surrogate_density(m, t)
    else:
        raise ValueError(f"unknown forward model {mode!r}; known: {FORWARD_MODES}")

    if cell_idx is None:
        return full
    idx = np.asarray(cell_idx)
    return lambda m: full(m)[idx]


def build_observations(obs_every: int = 15, margin: int = 4, *,
                       truth: np.ndarray | None = None, t: float = T_FINAL,
                       obs_error: float = OBS_ERROR,
                       seed: int | None = None,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize noisy density observations in Sod region interiors."""
    truth = TRUTH if truth is None else np.asarray(truth, dtype=float)
    exact_rho = exact_density_on_cells(truth, t, N_FIELD)
    features = shock_features(truth, t)

    mask = np.zeros(N_FIELD, dtype=bool)
    mask[::obs_every] = True
    for fx in features:
        c0 = int(np.clip(round(fx * N_FIELD - 0.5), 0, N_FIELD - 1))
        lo, hi = max(0, c0 - margin), min(N_FIELD - 1, c0 + margin)
        mask[lo:hi + 1] = False
    cell_idx = np.flatnonzero(mask)

    m = cell_idx.size
    H = np.zeros((N_STATE, m))
    H[FIELD_OFFSET + cell_idx, np.arange(m)] = 1.0
    rng = np.random.default_rng(seed)
    obs = exact_rho[cell_idx] + rng.normal(0.0, obs_error, size=m)
    return H, obs, cell_idx, exact_rho


def load_ensemble(directory: Path | str = ENSEMBLE_DIR) -> np.ndarray:
    import h5py

    files = sorted(Path(directory).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 ensemble files in {directory}")

    X = np.zeros((N_STATE, len(files)))
    for j, path in enumerate(files):
        with h5py.File(path, "r") as f:
            rho = np.atleast_2d(np.asarray(f["rho"], dtype=float))
            t = np.atleast_1d(np.asarray(f["t"], dtype=float))
            X[0, j] = float(f.attrs["p_high"])
            X[1, j] = float(f.attrs["p_low"])
            X[2, j] = float(f.attrs["rho_high"])
            X[3, j] = float(f.attrs["rho_low"])
        X[N_PARAMS, j] = t[-1]
        X[FIELD_OFFSET:, j] = rho[-1]
    return X


def make_plots(res: dict, outdir: Path, label: str = "dakota") -> None:
    """Marginal histograms, traces, posterior field band, and residuals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    post, chain = res["post"], res["chain"]
    burn = res["burn"]
    truth = res["truth"]
    tag = label.upper()

    fig, axes = plt.subplots(2, N_PARAMS, figsize=(16, 7), constrained_layout=True)
    for k, name in enumerate(PARAM_NAMES):
        ax = axes[0, k]
        ax.hist(post[:, k], bins=40, color="tab:blue", alpha=0.75, density=True)
        ax.axvline(truth[k], color="black", lw=2.0, label="truth")
        ax.axvline(res["mean"][k], color="tab:red", lw=1.6, ls="--", label="post. mean")
        ax.axvspan(res["quantiles"][0, k], res["quantiles"][2, k],
                   color="tab:red", alpha=0.12, label="95% CI")
        ax.set_title(name)
        ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")

        axt = axes[1, k]
        axt.plot(chain[:, k], color="0.4", lw=0.4)
        axt.axvline(burn, color="tab:orange", lw=1.2, ls=":", label="burn-in")
        axt.axhline(truth[k], color="black", lw=1.2)
        axt.set_xlabel("MCMC step")
        if k == 0:
            axt.set_ylabel("trace")
            axt.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{tag} posterior over Sod ICs  (g = {res['forward']}, "
                 f"acc = {res['acc_rate']:.2f})")
    p1 = outdir / f"{label}_marginals.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {p1}")

    x = res["x_cells"]
    fig2, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax.fill_between(x, res["field_lo"], res["field_hi"], color="tab:blue",
                    alpha=0.25, label="95% posterior band")
    ax.plot(x, res["field_mean"], color="tab:blue", lw=1.8,
            label=f"posterior mean (RMSE {res['post_rmse']:.2e})")
    ax.plot(x, res["exact"], color="black", lw=2.0, zorder=0, label="Exact")
    ax.scatter(res["obs_x"], res["obs_y"], color="gray", marker="o", s=16,
               edgecolor="white", linewidth=0.4, zorder=5, label="Observations")
    ax.set_ylabel("density rho")
    ax.set_title(f"{tag} reconstructed field  (g = {res['forward']}, t = {T_FINAL:.4f})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    axr.axhline(0.0, color="black", lw=0.8)
    axr.plot(x, res["field_mean"] - res["exact"], color="tab:blue", lw=1.4,
             label="posterior mean - exact")
    axr.scatter(res["obs_x"], np.zeros_like(res["obs_x"]), color="gray",
                marker="|", s=40, zorder=4)
    axr.set_xlabel("x")
    axr.set_ylabel("residual")
    axr.legend(loc="upper right", fontsize=9)
    axr.grid(True, alpha=0.3)
    p2 = outdir / f"{label}_field.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"[plot] wrote {p2}")
