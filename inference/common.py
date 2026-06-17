"""Shared infrastructure for the shock-tube parameter-inference scripts.

Every script under ``inference/`` (EnKF, cycled EnKF, ES-MDA, RW-MCMC, HMC)
solves the SAME inverse problem: infer the 4 Sod initial-condition parameters

    m = [p_high, p_low, rho_high, rho_low]

from synthetic density observations at ``T_FINAL``, drawn from the exact Sod
solution at the TRUTH operating point. This module owns everything those
scripts share, so each one stays a thin wrapper around its own algorithm:

  - problem constants (TRUTH, T_FINAL, N_FIELD, the augmented-state layout);
  - the FORWARD-MODEL FACTORY ``make_forward(mode, ...)`` dispatching on
    ``exact | euler | surrogate`` -- adding a new forward model means touching
    exactly this one place;
  - the observation builder (region-interior cells, never on a discontinuity);
  - the prior helpers (``load_ensemble`` from the 20 HDF5 files,
    ``forecast_ensemble`` drawn on the fly, ``prior_bounds`` for the samplers);
  - the bridge to the compiled C++ EnKF analysis step (``enkf_filter_cpp``);
  - the shared exact / prior-ensemble / posterior-mean comparison figure.

Forward models
--------------
``exact``     analytic Sod solution (fast; the same model the observations come
              from, i.e. an inverse crime -- fine for method development).
``euler``     numerical MUSCL-HLLC solver (honest: smears the fronts, so the
              inverse problem is solved across a real model discrepancy).
``surrogate`` the trained GPR surrogate (campaign output). Loaded lazily on
              first use; maps [p_high, p_low, rho_high, rho_low, t] -> rho(x).
"""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# This module lives one level under the repo root (inference/).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_simulations.Shock_Tube.sod_euler import EulerSolver1D  # noqa: E402
from task_simulations.Shock_Tube.sod_exact import (  # noqa: E402
    exact_density_on_cells,
    exact_state,
    shock_features,
)

__all__ = [
    "GAMMA", "N_PARAMS", "N_FIELD", "FIELD_OFFSET", "N_STATE", "T_FINAL",
    "TRUTH", "PARAM_NAMES", "PRIOR_MEAN", "PRIOR_SPREAD", "OBS_ERROR",
    "ENSEMBLE_DIR", "FORWARD_MODES",
    "exact_density_on_cells", "exact_state", "shock_features",
    "euler_density", "surrogate_density", "load_surrogate", "make_forward",
    "build_observations", "load_ensemble", "forecast_ensemble", "prior_bounds",
    "enkf_filter_cpp", "plot_field_comparison",
]

# --- Problem layout (matches EnKF/enkf_step.cpp & build_ensemble.cpp) -------
GAMMA = 1.4
N_PARAMS = 4                       # p_high, p_low, rho_high, rho_low
N_FIELD = 256                      # density cells
FIELD_OFFSET = N_PARAMS + 1        # density starts at row 5 (after 4 ICs + t)
N_STATE = FIELD_OFFSET + N_FIELD   # 261
T_FINAL = 6.0e-4                   # shock-tube final snapshot time

PARAM_NAMES = ["p_high", "p_low", "rho_high", "rho_low"]

# --- Truth the observations are synthesized from --- EASY TO CHANGE --------
TRUTH = np.array([1.0e5, 1.0e4, 1.0, 0.125])

# --- Forecast prior ---------------------------------------------------------
# Per-parameter relative spreads of the original 20-file ensemble
# (p ~ 4-5%, rho ~ 7-8%); members draw mean * (1 + spread * N(0,1)).
PRIOR_MEAN = TRUTH.copy()
PRIOR_SPREAD = np.array([0.04, 0.045, 0.08, 0.074])

OBS_ERROR = 0.01                   # default observation noise std

# 20-member HDF5 ensemble (prior for ES-MDA / cycled EnKF / sampler bounds).
ENSEMBLE_DIR = ROOT / "training_data" / "shock_tube" / "enkf_ensemble_files"


# --------------------------------------------------------------------------- #
# Forward models g(m) -> density field, and the factory that dispatches them.
# --------------------------------------------------------------------------- #
FORWARD_MODES = ("exact", "euler", "surrogate")

# Default campaign surrogate: [p_high, p_low, rho_high, rho_low, t] -> rho(x).
SURROGATE_PKL = ROOT / "training_runs" / "shock_tube" / "run_200" / "wf_0" / "surrogate.pkl"

_surrogate_cache: dict[Path, object] = {}


def euler_density(params, t: float = T_FINAL) -> np.ndarray:
    """Numerical Sod density at time t via the 1D MUSCL-HLLC Euler solver."""
    p = np.asarray(params, dtype=float)
    solver = EulerSolver1D(nx=N_FIELD, xmin=0.0, xmax=1.0, gamma=GAMMA, cfl=0.5)
    solver.set_sod_like(rho_high=p[2], p_high=p[0], rho_low=p[3], p_low=p[1], x0=0.5)
    solver.step_to(t)
    return solver.U[0]


def load_surrogate(path: Path | str | None = None):
    """Load (and cache) the trained GPR surrogate from a campaign pickle.

    Older pickles reference a ``surrogate.*`` module path; alias it onto the
    current ``task_train`` package before unpickling.
    """
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
    """GPR-surrogate Sod density at time t.

    ``Surrogate.predict`` returns the full solution vector in physical units,
    with the sol_keys stacked as N_FIELD-wide blocks in campaign order
    (rho, momentum, energy); only the rho block is returned here.
    """
    sur = load_surrogate(pkl)
    X = np.concatenate([np.asarray(params, dtype=float), [float(t)]])[None, :]
    Y, _ = sur.predict(X)
    return np.asarray(Y, dtype=float)[0, :N_FIELD]


def make_forward(mode: str, cell_idx: np.ndarray | None = None, t: float = T_FINAL):
    """Return ``g(m)`` -> density for the forward model ``mode``.

    ``mode`` is one of ``exact | euler | surrogate``. With ``cell_idx`` the
    callable returns only the observed cells (length ``m``), otherwise the full
    N_FIELD field. The surrogate is loaded once, here, so the returned callable
    is cheap.
    """
    if mode == "exact":
        full = lambda m: exact_density_on_cells(m, t, N_FIELD)
    elif mode == "euler":
        full = lambda m: euler_density(m, t)
    elif mode == "surrogate":
        load_surrogate()                      # fail fast + warm the cache
        full = lambda m: surrogate_density(m, t)
    else:
        raise ValueError(f"unknown forward model {mode!r}; known: {FORWARD_MODES}")

    if cell_idx is None:
        return full
    idx = np.asarray(cell_idx)
    return lambda m: full(m)[idx]


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
# Priors: the 20-file HDF5 ensemble, on-the-fly forecasts, and sampler bounds.
# --------------------------------------------------------------------------- #
def load_ensemble(directory: Path | str = ENSEMBLE_DIR) -> np.ndarray:
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


def forecast_ensemble(rng: np.random.Generator, ne: int, *,
                      prior_mean=PRIOR_MEAN, prior_spread=PRIOR_SPREAD,
                      forward: str = "euler", t: float = T_FINAL,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Draw an (N_STATE x ne) prior ensemble by forward-solving perturbed ICs.

    Each member draws its 4 ICs as ``prior_mean * (1 + prior_spread * N(0,1))``
    and is pushed to ``t`` by the ``forward`` model (euler/exact/surrogate);
    the resulting density is its local state, the ICs and t the augmented
    globals. Returns ``(X, member_params)`` with member_params (ne x N_PARAMS).
    """
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_spread = np.asarray(prior_spread, dtype=float)
    field_fn = make_forward(forward, t=t)

    X = np.zeros((N_STATE, ne))
    member_params = np.empty((ne, N_PARAMS))
    for j in range(ne):
        p = prior_mean * (1.0 + prior_spread * rng.standard_normal(N_PARAMS))
        X[:N_PARAMS, j] = p
        X[N_PARAMS, j] = t
        X[FIELD_OFFSET:, j] = field_fn(p)
        member_params[j] = p
    return X, member_params


def prior_bounds(directory: Path | str = ENSEMBLE_DIR, pad: float = 0.25,
                 contain: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-prior support for the samplers: ensemble [min, max] per param,
    padded by ``pad`` (fraction of the half-width).

    If ``contain`` (e.g. the truth) falls outside that box, the box is expanded
    so it sits inside with the same fractional margin -- otherwise a shifted
    truth could lie in a zero-prior region the sampler can never reach.
    """
    M = load_ensemble(directory)[:N_PARAMS, :]
    lo, hi = M.min(axis=1), M.max(axis=1)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    lo, hi = mid - (1.0 + pad) * half, mid + (1.0 + pad) * half
    if contain is not None:
        margin = (1.0 + pad) * half
        lo = np.minimum(lo, contain - margin)
        hi = np.maximum(hi, contain + margin)
    return lo, hi


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
