"""Global common layer for the Sod shock-tube inverse problem.

Every parameter-estimation driver in the repo -- the ensemble work in ``EnKF/``,
the samplers in ``MCMC/`` and the Dakota calibration in ``dakota_mcmc/`` -- solves
the SAME inverse problem: infer the 4 Sod initial-condition parameters

    m = [p_high, p_low, rho_high, rho_low]

from noisy density observations at ``T_FINAL`` drawn from the EXACT Sod solution
at the TRUTH operating point. This module is the single source of truth for
everything those folders used to each redefine:

  - the problem constants / augmented-state layout (TRUTH, T_FINAL, N_FIELD, ...);
  - the observation builder ``build_observations`` (region interiors, never a
    discontinuity);
  - the forward models g(m) -> density (``_forward_npz`` shelling out to the Sod
    CLIs, the in-process Euler/exact shortcuts, the GPR surrogate) and the
    ``make_forward`` factory dispatching ``exact | euler | surrogate``.

The per-folder ``*_common`` modules import from here and add only what is unique
to their algorithm (C++ EnKF bridge, sampler priors, package-specific plots).
The only outside dependency is the physics stack in ``task_simulations/Shock_Tube``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# This module lives one level under the repo root (helpers/); make the physics
# solvers importable whether a driver runs from its folder or the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_simulations.Shock_Tube.sod_exact import (  # noqa: E402
    exact_density_on_cells,
    shock_features,
)

# --- Problem layout (matches EnKF/enkf_step.cpp & build_ensemble.cpp) -------
# Augmented state (261 rows): the 4 Sod ICs + time stacked on the 256-cell
# density  [ p_high, p_low, rho_high, rho_low, t,  rho_0 ... rho_255 ].
GAMMA = 1.4                        # ideal-gas ratio for the Euler solver
N_PARAMS = 4                       # p_high, p_low, rho_high, rho_low
N_FIELD = 256                      # density cells
FIELD_OFFSET = N_PARAMS + 1        # density starts at row 5 (after 4 ICs + t)
N_STATE = FIELD_OFFSET + N_FIELD   # 261
T_FINAL = 6.0e-4                   # shock-tube final snapshot time

PARAM_NAMES = ["p_high", "p_low", "rho_high", "rho_low"]

# --- Truth the observations are synthesized from --- EASY TO CHANGE --------
TRUTH = np.array([1.0e5, 1.0e4, 1.0, 0.125])
DEFAULT_DATA_POINT = list(TRUTH)   # mean the generated EnKF ensemble perturbs around

OBS_ERROR = 0.01                   # default observation noise std (likelihood sigma)

# 20-member HDF5 ensemble (prior for ES-MDA / cycled EnKF / sampler bounds).
ENSEMBLE_DIR = ROOT / "training_data" / "shock_tube" / "enkf_ensemble_files"

# --- Forward models ---------------------------------------------------------
FORWARD_MODES = ("exact", "euler", "surrogate")

# Default campaign surrogate: [p_high, p_low, rho_high, rho_low, t] -> rho(x).
SURROGATE_PKL = ROOT / "training_runs" / "shock_tube" / "run_200" / "wf_0" / "surrogate.pkl"

_surrogate_cache: dict[Path, object] = {}

# Standalone Sod CLIs the per-sample forward models shell out to.
_SOD_EXACT = ROOT / "task_simulations" / "Shock_Tube" / "sod_exact.py"
_SOD_EULER = ROOT / "task_simulations" / "Shock_Tube" / "sod_euler.py"

# ONE scratch .npz per process: every exact/euler forward call writes rho(x) here
# via the CLI's --out and reads it straight back, OVERWRITING the previous sample
# (keyed by PID so concurrent runs don't clobber each other).
_SCRATCH_NPZ = Path(tempfile.gettempdir()) / f"sod_forward_{os.getpid()}.npz"


# --------------------------------------------------------------------------- #
# Forward models g(m) -> density field, and the factory that dispatches them.
# --------------------------------------------------------------------------- #
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
    EXCLUDED, so observations sit where the values are well-determined. Observing
    on a front instead injects huge, ambiguous innovations (each ensemble
    member's shock sits somewhere else), which drives overshoot.

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
# Prior ensemble: read the HDF5 checkpoints into one augmented-state matrix.
# --------------------------------------------------------------------------- #
def load_ensemble(directory: Path | str = ENSEMBLE_DIR) -> np.ndarray:
    """Load the HDF5 ensemble into an (N_STATE x Ne) augmented-state matrix.

    Each file is a sim-format trajectory (rho/momentum/energy shaped
    (n_snap, nx) + the 4 IC attrs); the LAST snapshot supplies the local density
    state and its time the augmented ``t`` row.
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
