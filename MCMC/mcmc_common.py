"""Shared infrastructure for the self-contained MCMC samplers in ``MCMC/``.

Both samplers in this folder -- ``mh_mcmc.py`` (random-walk Metropolis) and
``ham_mcmc.py`` (HMC) -- solve the SAME inverse problem: sample the posterior
over the 4 Sod initial-condition parameters

    m = [p_high, p_low, rho_high, rho_low]

from density observations at ``T_FINAL`` drawn from the exact Sod solution at the
TRUTH operating point. Everything they share lives here, so the folder stands on
its own (the only outside dependency is the physics solvers in
``task_simulations/Shock_Tube/``):

  - problem constants (TRUTH, T_FINAL, N_FIELD, PARAM_NAMES);
  - the forward-model factory ``make_forward(mode, ...)`` dispatching on
    ``exact | euler | surrogate``;
  - the observation builder (region-interior cells, never on a discontinuity);
  - the prior helpers (``load_ensemble``, ``prior_bounds`` for the samplers);
  - the shared marginal/trace/field figure (``make_plots``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# This module lives one level under the repo root (MCMC/); make the physics
# solvers importable whether we run from MCMC/ or the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# exact_density_on_cells / shock_features build the one-time TRUTH observations
# in-process; the per-sample forward models below instead shell out to the
# standalone Sod CLIs (sod_exact.py / sod_euler.py) via subprocess.
from task_simulations.Shock_Tube.sod_exact import (  # noqa: E402
    exact_density_on_cells,
    shock_features,
)

# --- Problem layout --------------------------------------------------------
N_PARAMS = 4                       # p_high, p_low, rho_high, rho_low
N_FIELD = 256                      # density cells
FIELD_OFFSET = N_PARAMS + 1        # density starts at row 5 (after 4 ICs + t)
N_STATE = FIELD_OFFSET + N_FIELD   # 261
T_FINAL = .0006                 # shock-tube final snapshot time

PARAM_NAMES = ["p_high", "p_low", "rho_high", "rho_low"]

# --- Truth the observations are synthesized from --- EASY TO CHANGE --------
TRUTH = np.array([1.0e5, 0.1e5, 1.0, 0.125])

OBS_ERROR = 0.01                   # default observation noise std (likelihood sigma)

# 20-member HDF5 ensemble used only to set the uniform-prior box (prior_bounds).
ENSEMBLE_DIR = ROOT / "training_data" / "shock_tube" / "enkf_ensemble_files"


# --------------------------------------------------------------------------- #
# Forward models g(m) -> density field, and the factory that dispatches them.
# --------------------------------------------------------------------------- #
FORWARD_MODES = ("exact", "euler", "surrogate")

# Default campaign surrogate: [p_high, p_low, rho_high, rho_low, t] -> rho(x).
SURROGATE_PKL = ROOT / "training_runs" / "shock_tube" / "run_200" / "wf_0" / "surrogate.pkl"

_surrogate_cache: dict[Path, object] = {}

GAMMA = 1.4                        # only the in-process Euler shortcut needs this

# Standalone Sod CLIs the per-sample forward models shell out to.
_SOD_EXACT = ROOT / "task_simulations" / "Shock_Tube" / "sod_exact.py"
_SOD_EULER = ROOT / "task_simulations" / "Shock_Tube" / "sod_euler.py"

# ONE scratch .npz per process: every exact/euler forward call writes rho(x) here
# via the CLI's --out and reads it straight back, OVERWRITING the previous sample
# (keyed by PID so concurrent sampler runs don't clobber each other).
_SCRATCH_NPZ = Path(tempfile.gettempdir()) / f"mcmc_forward_{os.getpid()}.npz"


def _forward_npz(script: Path, params, t: float, out_path: Path = _SCRATCH_NPZ) -> np.ndarray:
    """Run a standalone Sod CLI to write rho(x) at time t into out_path (.npz),
    overwriting it, then read the density field back.

    The CLI takes ``p_high p_low rho_high rho_low t`` and ``--out FILE.npz``
    (which stores x/rho/u/p/t/params); we return the ``rho`` field.
    """
    p = np.asarray(params, dtype=float)
    cmd = [sys.executable, str(script), *(f"{v:.10g}" for v in p), f"{t:.10g}",
           "--nx", str(N_FIELD), "--out", str(out_path), "--quiet", "--fast"]
    subprocess.run(cmd, check=True)
    with np.load(out_path) as data:
        return np.asarray(data["rho"], dtype=float)


def _euler_in_process(params, t: float) -> np.ndarray:
    """In-process MUSCL-HLLC Euler density -- the fast testing shortcut."""
    from task_simulations.Shock_Tube.sod_euler import EulerSolver1D
    p = np.asarray(params, dtype=float)
    solver = EulerSolver1D(nx=N_FIELD, xmin=0.0, xmax=1.0, gamma=GAMMA, cfl=0.5)
    solver.set_sod_like(rho_high=p[2], p_high=p[0], rho_low=p[3], p_low=p[1], x0=0.5)
    solver.step_to(t)
    return solver.U[0]


def exact_density(params, t: float = T_FINAL, *, in_process: bool = False) -> np.ndarray:
    """Analytic Sod density at time t.

    Default: shell out to the sod_exact.py CLI (--out npz), the path that mirrors
    the eventual CFD executable. ``in_process=True`` is the fast shortcut for
    testing -- same analytic solution, no subprocess.
    """
    if in_process:
        return exact_density_on_cells(params, t, N_FIELD)
    return _forward_npz(_SOD_EXACT, params, t)


def euler_density(params, t: float = T_FINAL, *, in_process: bool = False) -> np.ndarray:
    """Numerical Sod density at time t.

    Default: shell out to the sod_euler.py CLI (--out npz). ``in_process=True``
    is the fast shortcut for testing (same solver, called directly).
    """
    if in_process:
        return _euler_in_process(params, t)
    return _forward_npz(_SOD_EULER, params, t)


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
    """GPR-surrogate Sod density at time t (rho block of the prediction)."""
    sur = load_surrogate(pkl)
    X = np.concatenate([np.asarray(params, dtype=float), [float(t)]])[None, :]
    Y, _ = sur.predict(X)
    return np.asarray(Y, dtype=float)[0, :N_FIELD]


def make_forward(mode: str, cell_idx: np.ndarray | None = None, t: float = T_FINAL,
                 *, in_process: bool = False):
    """Return ``g(m)`` -> density for the forward model ``mode``.

    ``mode`` is one of ``exact | euler | surrogate``. With ``cell_idx`` the
    callable returns only the observed cells (length ``m``), otherwise the full
    N_FIELD field. ``in_process=True`` runs the exact/euler solvers directly
    instead of shelling out to their CLIs -- a fast shortcut for testing (the
    subprocess default mirrors the eventual CFD-executable forward model). The
    surrogate is always in-process, so the flag is a no-op there.
    """
    if mode == "exact":
        full = lambda m: exact_density(m, t, in_process=in_process)
    elif mode == "euler":
        full = lambda m: euler_density(m, t, in_process=in_process)
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
# Prior: the 20-file HDF5 ensemble and the padded uniform-prior box.
# --------------------------------------------------------------------------- #
def load_ensemble(directory: Path | str = ENSEMBLE_DIR) -> np.ndarray:
    """Load the HDF5 ensemble into an (N_STATE x Ne) augmented-state matrix.

    Each file is a sim-format trajectory (rho/momentum/energy + the 4 IC attrs);
    the LAST snapshot supplies the local density state and its time the augmented
    ``t`` row. The samplers use only the top N_PARAMS rows.
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


def prior_bounds(directory: Path | str = ENSEMBLE_DIR, frac: float = 0.25,
                 contain: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-prior support for the samplers: each parameter's default value
    +/- a relative fraction ``frac``, i.e. ``center * (1 -/+ frac)`` per param.

    With ``frac=0.25`` and the default ``p_high=1e5`` this gives ``[75000,
    125000]``; ``rho_high=1.0`` gives ``[0.75, 1.25]``; and so on. The center is
    ``contain`` when supplied (e.g. the truth the samplers pass), else ``TRUTH``.
    ``directory`` is retained for signature compatibility and is unused.
    """
    center = TRUTH if contain is None else np.asarray(contain, dtype=float)
    lo = center * (1.0 - frac)
    hi = center * (1.0 + frac)
    return lo, hi


# --------------------------------------------------------------------------- #
# Plots: marginal histograms, traces, and the reconstructed field band.
# Shared by mh_mcmc.py and ham_mcmc.py; ``label`` tags titles/filenames.
# --------------------------------------------------------------------------- #
def make_plots(res: dict, outdir: Path, label: str = "mcmc") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    post, chain = res["post"], res["chain"]
    burn = res["burn"]
    truth = res["truth"]
    tag = label.upper()

    # --- marginals + traces (2 rows x 4 cols) -------------------------------
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
    print(f"[plot] wrote {p1}")

    # --- reconstructed density field with 95% band -------------------------
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
    print(f"[plot] wrote {p2}")
