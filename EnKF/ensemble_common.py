"""Shared infrastructure for the self-contained EnKF / ensemble work in ``EnKF/``.

Everything the shock-tube ensemble drivers in this folder need lives here, so the
folder stands on its own (the only outside dependency is the physics solvers in
``task_simulations/Shock_Tube/``):

  - problem constants (TRUTH, T_FINAL, N_FIELD, the augmented-state layout);
  - the observation builder (region-interior cells, never on a discontinuity);
  - the prior helpers (``gen_ensemble`` forward-solves perturbed ICs to disk;
    ``load_ensemble`` reads the resulting HDF5 files into an augmented state);
  - the bridge to the compiled C++ EnKF analysis step (``enkf_filter_cpp``);
  - the shared exact / prior-ensemble / posterior-mean comparison figure.

Augmented state (261 rows): the 4 Sod ICs + time stacked on the 256-cell density

    [ p_high, p_low, rho_high, rho_low, t,  rho_0 ... rho_255 ]
     |<------------ 5 globals ----------->|  |<-- 256-cell density -->|
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# This module lives one level under the repo root (EnKF/); make the physics
# solvers importable whether we run from EnKF/ or the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_simulations.Shock_Tube.sod_exact import (  # noqa: E402
    exact_density_on_cells,
    shock_features,
)

# --- Problem layout (matches EnKF/enkf_step.cpp) ---------------------------
N_PARAMS = 4                       # p_high, p_low, rho_high, rho_low
N_FIELD = 256                      # density cells
FIELD_OFFSET = N_PARAMS + 1        # density starts at row 5 (after 4 ICs + t)
N_STATE = FIELD_OFFSET + N_FIELD   # 261
T_FINAL = 6.0e-4                   # shock-tube final snapshot time

PARAM_NAMES = ["p_high", "p_low", "rho_high", "rho_low"]

# --- Truth the observations are synthesized from --- EASY TO CHANGE --------
TRUTH = np.array([1.0e5, 1.0e4, 1.0, 0.125])
DEFAULT_DATA_POINT = list(TRUTH)   # mean the generated ensemble perturbs around

OBS_ERROR = 0.01                   # default observation noise std


# --------------------------------------------------------------------------- #
# Observations: exact Sod density in the 5 region INTERIORS, never on a front.
# --------------------------------------------------------------------------- #
def build_observations(obs_every: int = 15, margin: int = 4, *,
                       truth: np.ndarray | None = None, t: float = T_FINAL,
                       obs_error: float = OBS_ERROR,
                       seed: int | None = None,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize observations from the exact solution at time ``t``, placed
    ONLY in the interiors of the 5 Sod regions -- never on the discontinuities.

    A baseline of every ``obs_every``-th cell is taken, then every cell within
    ``margin`` of any feature (rarefaction head/tail, contact, shock) is
    EXCLUDED, so observations sit where the values are well-determined.
    Observing on a front instead injects huge, ambiguous innovations (each
    ensemble member's shock sits somewhere else), which drives overshoot.

    At each observed cell the data are ``exact_rho + N(0, obs_error^2)``, matching
    the Gaussian likelihood assumed by the samplers and Dakota calibration.

    Returns ``(H, obs, cell_idx, exact_rho)`` with H the (N_STATE x m)
    augmented-state observation operator and ``exact_rho`` the unperturbed field.
    """
    truth = TRUTH if truth is None else np.asarray(truth, dtype=float)
    exact_rho = exact_density_on_cells(truth, t, N_FIELD)
    features = shock_features(truth, t)

    mask = np.zeros(N_FIELD, dtype=bool)
    mask[::obs_every] = True                                  # baseline coverage
    for fx in features:
        c0 = int(np.clip(round(fx * N_FIELD - 0.5), 0, N_FIELD - 1))
        lo, hi = max(0, c0 - margin), min(N_FIELD - 1, c0 + margin)
        mask[lo:hi + 1] = False                               # keep clear of the jump
    cell_idx = np.flatnonzero(mask)

    m = cell_idx.size
    H = np.zeros((N_STATE, m))
    H[FIELD_OFFSET + cell_idx, np.arange(m)] = 1.0
    rng = np.random.default_rng(seed)
    obs = exact_rho[cell_idx] + rng.normal(0.0, obs_error, size=m)
    return H, obs, cell_idx, exact_rho


# --------------------------------------------------------------------------- #
# Record the estimated global parameters from an analysis step to disk.
# --------------------------------------------------------------------------- #
def record_parameters(prior_ensemble, post_ensemble, save_path, *,
                      truth=TRUTH, param_names=PARAM_NAMES) -> dict:
    """Write the inferred global parameters from an EnKF analysis to JSON.

    Takes the augmented prior and analysis ensembles (only their top ``N_PARAMS``
    rows are used) and records, per parameter, the prior vs analysis mean and std,
    the truth, and the analysis-mean relative error -- plus the full per-member
    analysis parameter values -- so the parameter estimate is preserved next to
    the figure. Returns the recorded dict.
    """
    prior_p = np.asarray(prior_ensemble, dtype=float)[:N_PARAMS, :]
    post_p = np.asarray(post_ensemble, dtype=float)[:N_PARAMS, :]
    truth = np.asarray(truth, dtype=float)

    post_mean = post_p.mean(axis=1)
    record = {
        "param_names": list(param_names),
        "truth": truth.tolist(),
        "prior_mean": prior_p.mean(axis=1).tolist(),
        "prior_std": prior_p.std(axis=1).tolist(),
        "analysis_mean": post_mean.tolist(),
        "analysis_std": post_p.std(axis=1).tolist(),
        "rel_error": (np.abs(post_mean - truth) / np.abs(truth)).tolist(),
        "n_members": int(post_p.shape[1]),
        "analysis_members": post_p.tolist(),     # (N_PARAMS x Ne)
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[record] wrote estimated parameters to {save_path}")
    return record


# --------------------------------------------------------------------------- #
# Prior ensemble: forward-solve perturbed ICs to disk, then read them back.
# --------------------------------------------------------------------------- #
_SOD_EULER = ROOT / "task_simulations" / "Shock_Tube" / "sod_euler.py"


def run_euler_shock_tube(params, out_dir: Path | str, *, t: float = T_FINAL):
    """Forward-solve one Sod member to ``t`` and write a sim-format HDF5 into
    ``out_dir`` (auto-named). ``params`` is the 4-vector [p_high, p_low,
    rho_high, rho_low]. Raises if the solver fails.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # CLI: p_high p_low rho_high rho_low t [--h5 DIR/]  (trailing / => auto-name)
    cmd = [
        sys.executable, str(_SOD_EULER),
        *map(str, params), str(t),
        "--h5", f"{out_dir}{os.sep}",
    ]
    subprocess.run(cmd, check=True)


def gen_ensemble(n_members: int, out_dir: Path | str, *,
                 spread: float = 0.1, defaults=DEFAULT_DATA_POINT,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate an ``n_members`` prior ensemble on disk and return it loaded.

    Each member perturbs ``defaults`` by ``mean * (1 + spread * N(0,1))`` and is
    forward-solved to ``T_FINAL``. Returns the (N_STATE x n_members) augmented
    state read back from the freshly written HDF5 files.
    """
    rng = np.random.default_rng() if rng is None else rng
    out_dir = Path(out_dir)
    for _ in range(n_members):
        params = [d * (1.0 + rng.normal(0.0, spread)) for d in defaults]
        run_euler_shock_tube(params, out_dir)
    return load_ensemble(out_dir)


def load_ensemble(directory: Path | str) -> np.ndarray:
    """Load the HDF5 ensemble into an (N_STATE x Ne) augmented-state matrix.

    Each file is a sim-format trajectory (rho/momentum/energy shaped
    (n_snap, nx) + the 4 IC attrs); the LAST snapshot supplies the local
    density state and its time the augmented ``t`` row.
    """
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


# --------------------------------------------------------------------------- #
# C++ root filter: one stochastic EnKF analysis step in Eigen.
# --------------------------------------------------------------------------- #
# Python builds the ensemble/obs/operator, hands them over a small
# little-endian binary bundle (column-major to match Eigen), and reads back the
# analysis ensemble. Layout matches EnKF/enkf_step.cpp.
_ENKF_STEP_SRC = ROOT / "EnKF" / "enkf_step.cpp"
_ENKF_STEP_BIN = ROOT / "EnKF" / "enkf_step"
_EIGEN_INC = "/usr/include/eigen3"


def _ensure_enkf_step() -> Path:
    """Compile EnKF/enkf_step if the binary is missing or older than its source."""
    if _ENKF_STEP_BIN.exists() and _ENKF_STEP_BIN.stat().st_mtime >= _ENKF_STEP_SRC.stat().st_mtime:
        return _ENKF_STEP_BIN
    cmd = ["g++", "-O3", "-std=c++17", f"-I{_EIGEN_INC}",
           str(_ENKF_STEP_SRC), "-o", str(_ENKF_STEP_BIN)]
    subprocess.run(cmd, check=True)
    return _ENKF_STEP_BIN


def enkf_filter_cpp(X, obs, H, obs_error, *, state_loc, obs_loc,
                    num_globals, loc_rad, localize, seed=-1, return_perturbed=False):
    """Run one C++ EnKF analysis step and return the analysis ensemble X^a.

    If ``return_perturbed`` is True, also return the perturbed observation
    ensemble y^o_e (m x Ne) the filter built -- as ``(X^a, y^o_e)`` -- so the
    caller can confirm observations are perturbed independently per member.
    """
    binary = _ensure_enkf_step()

    X = np.ascontiguousarray(X, dtype=np.float64)
    H = np.ascontiguousarray(H, dtype=np.float64)
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    state_loc = np.ascontiguousarray(state_loc, dtype=np.float64)
    obs_loc = np.ascontiguousarray(obs_loc, dtype=np.float64)

    n_state, ne = X.shape
    m = obs.size

    header = struct.pack(
        "<5iqdd", n_state, ne, m, int(num_globals), int(bool(localize)),
        int(seed), float(obs_error), float(loc_rad),
    )

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.bin"
        out_path = Path(td) / "out.bin"
        pobs_path = Path(td) / "pobs.bin"
        with open(in_path, "wb") as f:
            f.write(header)
            f.write(X.tobytes(order="F"))           # column-major for Eigen
            f.write(obs.tobytes())
            f.write(H.tobytes(order="F"))
            f.write(state_loc.tobytes())
            f.write(obs_loc.tobytes())

        cmd = [str(binary), str(in_path), str(out_path)]
        if return_perturbed:
            cmd.append(str(pobs_path))              # 4th arg => also dump y^o_e
        subprocess.run(cmd, check=True)

        data = np.fromfile(out_path, dtype=np.float64, count=n_state * ne)
        X_a = data.reshape((n_state, ne), order="F")
        if return_perturbed:
            pdata = np.fromfile(pobs_path, dtype=np.float64, count=m * ne)
            return X_a, pdata.reshape((m, ne), order="F")

    return X_a


# --------------------------------------------------------------------------- #
# Shared figure: exact (black) / prior ensemble (grey) / posterior mean (red)
# with an observation overlay and a residual panel.
# --------------------------------------------------------------------------- #
def plot_field_comparison(*, exact, members, post_mean, obs_x, obs_y,
                          post_rmse, title, post_label, save_path,
                          prior_label="Prior ensemble") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(N_FIELD) / N_FIELD
    prior = members.mean(axis=1)

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})

    # --- top: density field ---------------------------------------------
    ax.plot(x, exact, color="black", lw=2.0, zorder=0, label="Exact")
    for i in range(members.shape[1]):
        ax.plot(x, members[:, i], color="0.8", lw=0.8, zorder=1,
                label=prior_label if i == 0 else None)
    ax.plot(x, post_mean, color="tab:red", lw=1.9, zorder=4,
            label=f"{post_label} (RMSE {post_rmse:.2e})")
    ax.scatter(obs_x, obs_y, color="gray", marker="o", s=16,
               edgecolor="white", linewidth=0.4, zorder=5, label="Observations")
    ax.set_ylabel("density rho")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # --- bottom: residual vs exact (prior & posterior means) -------------
    ax_r.axhline(0.0, color="black", lw=0.8)
    ax_r.plot(x, prior - exact, color="tab:blue", lw=1.0, ls="--", label="prior mean")
    ax_r.plot(x, post_mean - exact, color="tab:red", lw=1.4, label="posterior mean")
    ax_r.scatter(obs_x, np.zeros_like(obs_x), color="gray",
                 marker="|", s=40, zorder=4)
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("residual")
    ax_r.legend(loc="upper right", framealpha=0.95, ncol=2, fontsize=9)
    ax_r.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")


# --------------------------------------------------------------------------- #
# Euler forward model (in-process) -- used to push a parameter vector to a
# density field for the parameter-recovery comparison figure.
# --------------------------------------------------------------------------- #
GAMMA = 1.4


def euler_density(params, t: float = T_FINAL, *, nx: int = N_FIELD) -> np.ndarray:
    """Numerical Sod density at time ``t`` via the 1D MUSCL-HLLC Euler solver."""
    from task_simulations.Shock_Tube.sod_euler import EulerSolver1D
    p = np.asarray(params, dtype=float)
    solver = EulerSolver1D(nx=nx, xmin=0.0, xmax=1.0, gamma=GAMMA, cfl=0.5)
    solver.set_sod_like(rho_high=p[2], p_high=p[0], rho_low=p[3], p_low=p[1], x0=0.5)
    solver.step_to(t)
    return solver.U[0]


# --------------------------------------------------------------------------- #
# Parameter-recovery figure: Euler density forward-solved from the EnKF-predicted
# params vs Euler from the TRUE params. Answers "do the recovered parameters
# reproduce the truth's field when pushed back through the forward model?"
# --------------------------------------------------------------------------- #
def plot_param_forward_compare(pred_params, save_path, *, prior_params=None,
                               truth=TRUTH, t: float = T_FINAL,
                               param_names=PARAM_NAMES,
                               title="Euler(predicted params) vs Euler(truth params)",
                               ) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_params = np.asarray(pred_params, dtype=float)[:N_PARAMS]
    truth = np.asarray(truth, dtype=float)[:N_PARAMS]

    rho_truth = euler_density(truth, t)
    rho_pred = euler_density(pred_params, t)
    rmse = float(np.sqrt(np.mean((rho_pred - rho_truth) ** 2)))

    rho_prior = None
    if prior_params is not None:
        prior_params = np.asarray(prior_params, dtype=float)[:N_PARAMS]
        rho_prior = euler_density(prior_params, t)
        prior_rmse = float(np.sqrt(np.mean((rho_prior - rho_truth) ** 2)))

    x = np.arange(N_FIELD) / N_FIELD

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax.plot(x, rho_truth, color="black", lw=2.0, zorder=1, label="Euler(truth params)")
    if rho_prior is not None:
        ax.plot(x, rho_prior, color="tab:blue", lw=1.5, ls="--", zorder=2,
                label=f"Euler(prior-mean params)  (RMSE {prior_rmse:.2e})")
    ax.plot(x, rho_pred, color="tab:red", lw=1.8, zorder=3,
            label=f"Euler(predicted params)  (RMSE {rmse:.2e})")
    ax.set_ylabel("density rho")
    ax.set_title(title)
    leg = ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # --- value table, same width / font / rounded frame as the legend ------
    from matplotlib.patches import FancyBboxPatch

    has_prior = rho_prior is not None
    cols = ["param", "truth"] + (["prior"] if has_prior else []) + ["predicted"]
    col_color = {"truth": "black", "prior": "tab:blue", "predicted": "tab:red"}
    values = {"truth": truth, "prior": prior_params if has_prior else None,
              "predicted": pred_params}

    fig.canvas.draw()                                  # realize legend geometry
    lb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    fp = leg.get_texts()[0].get_fontproperties()       # match the legend font

    n_rows = len(param_names) + 1
    row_h = 0.050
    height = row_h * n_rows
    pad_x = 0.012
    gap = 0.015
    top = lb.y0 - gap
    bot = top - height

    # Box = same width as the legend. The param name and the (short) truth column
    # take slim slices, leaving more room for the wider prior/predicted columns.
    width = lb.width
    bx0 = lb.x0
    param_frac = 0.27
    truth_frac = 0.20
    other_frac = (1.0 - param_frac - truth_frac) / max(len(cols) - 2, 1)
    redge = {}                                         # right edge (frac of width) per value col
    acc = param_frac
    for j in range(1, len(cols)):
        acc += truth_frac if j == 1 else other_frac
        redge[j] = acc

    # Solid rounded frame like the first iteration's text box: white fill over
    # the grid, light-gray edge. mutation_aspect (axes pixel w/h) keeps the
    # corner radius circular despite the axes-fraction coordinates.
    axbb = ax.get_window_extent()
    box = FancyBboxPatch(
        (bx0, bot), width, height, transform=ax.transAxes,
        boxstyle="round,pad=0,rounding_size=0.005",
        mutation_aspect=axbb.width / axbb.height,
        facecolor="white", edgecolor="0.7", linewidth=1.0,
        zorder=10, clip_on=False)
    ax.add_patch(box)

    def _cell(j, r, text, color):
        y = top - (r + 0.5) * row_h
        if j == 0:                                     # param names: left edge
            ax.text(bx0 + pad_x, y, text, transform=ax.transAxes,
                    fontproperties=fp, color=color, ha="left", va="center",
                    zorder=11)
        else:                                          # values: right edge of col
            x = bx0 + redge[j] * width - pad_x
            ax.text(x, y, text, transform=ax.transAxes,
                    fontproperties=fp, color=color, ha="right", va="center",
                    zorder=11)

    for j, c in enumerate(cols):                       # header row (colored dots)
        _cell(j, 0, c if j == 0 else f"● {c}", col_color.get(c, "black"))
    for k, nm in enumerate(param_names):               # one row per parameter
        _cell(0, k + 1, nm, "black")
        for j, c in enumerate(cols[1:], start=1):
            _cell(j, k + 1, f"{values[c][k]:.4g}", "black")

    ax_r.axhline(0.0, color="black", lw=0.8)
    if rho_prior is not None:
        ax_r.plot(x, rho_prior - rho_truth, color="tab:blue", lw=1.0, ls="--",
                  label="prior-mean - truth")
    ax_r.plot(x, rho_pred - rho_truth, color="tab:red", lw=1.4,
              label="predicted - truth")
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("residual")
    ax_r.legend(loc="upper right", fontsize=9)
    ax_r.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}  (param-recovery RMSE {rmse:.4e})")
    return rmse
