"""Generate an EnKF-ready ensemble of shock-tube runs.

Each ensemble member is one shock-tube simulation whose initial conditions are
log-normally perturbed around a nominal operating point (same scheme as
``gen_ensemble.py``). The member's *state vector* is laid out exactly as the EnKF
expects, with the augmented (global) parameters stacked on top of the local state
field::

    column = [ p_high, p_low, rho_high, rho_low, t_f,  rho_0 ... rho_255 ]
             |<------------ 4 ICs ----------->| time  |<-- 256-cell density -->|

so the full ensemble is a ``(n_state=261, ens_size)`` matrix, one member per
column — the convention ``EnKF::filter`` reads. ``t_f`` is the (static) final
snapshot time, identical across members.

A sparse density observation operator is built from an unperturbed *truth* run:
``H`` selects every ``--obs-every``-th density cell, ``y^o`` is the truth density
at those cells, and ``obs_error`` is the measurement std. ``H`` follows the EnKF's
convention where ``H^T @ state`` maps a state vector into observation space, i.e.
``H`` has shape ``(n_state, m)`` with a single 1 per column at the observed row.

The per-member sims are written as standard shock-tube ``.h5`` files (same
datasets/attrs as ``run_shock_tube``), so they double as ROSE training data via
``ParameterSpace.construct_X`` / ``construct_Y``.

Note: this is the bootstrap source of the ensemble. The eventual plan is to
compose the ensemble from a ROSE run's ``data_dir`` plus supplemental members
drawn from the trained surrogate; this script stands in until that path exists.

Run from the repo root::

    python task_simulations/Shock_Tube/gen_enkf_ensemble.py --N 30 --obs-every 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# This script lives two levels under the repo root, so the shared
# ensure_project_root (which assumes one level) does not apply here.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task_simulations.Shock_Tube.run_shock_tube import run_shock_tube

# Parameter order matches ParameterSpace("shock_tube").param_names; the user's
# (p_L, p_R, rho_L, rho_R) map onto these since run_shock_tube sets
# p_L=p_high, p_R=p_low, rho_L=rho_high, rho_R=rho_low.
PARAM_NAMES = ("p_high", "p_low", "rho_high", "rho_low")
NOMINAL = np.array([1.0e5, 1.0e4, 1.0, 0.125])  # canonical shock-tube operating point

N_PARAMS = len(PARAM_NAMES)   # 4 global (augmented) parameters
TIME_INDEX = N_PARAMS         # row 4 holds t_f
FIELD_OFFSET = N_PARAMS + 1   # density field starts at row 5
FIELD_KEY = "rho"


def sample_params(rng: np.random.Generator, p_sig: float, rho_sig: float) -> np.ndarray:
    """Log-normal multiplicative perturbation of the four ICs (cf. gen_ensemble.py)."""
    z = rng.standard_normal(N_PARAMS)
    sig = np.array([p_sig, p_sig, rho_sig, rho_sig])  # pressures vs densities
    return NOMINAL * np.exp(sig * z)


def run_member(params: np.ndarray, out_h5: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Run one shock-tube sim, returning (rho_final[256], t_f, x[256])."""
    run_shock_tube(str(out_h5), *params.tolist())
    with h5py.File(out_h5, "r") as f:
        rho = np.asarray(f[FIELD_KEY], dtype=float)  # (n_snap, nx)
        t = np.asarray(f["t"], dtype=float)
        x = np.asarray(f["x"], dtype=float)
    return rho[-1], float(t[-1]), x


def state_vector(params: np.ndarray, t_f: float, rho_final: np.ndarray) -> np.ndarray:
    """Assemble [p_high, p_low, rho_high, rho_low, t_f, rho_0..rho_255]."""
    return np.concatenate([params, [t_f], rho_final])


def build_sparse_obs(truth_state: np.ndarray, n_field: int, every: int, obs_error: float):
    """Sparse density observation operator from the truth state.

    Returns (obs, H, obs_error, cell_idx, state_idx) where H has shape
    (n_state, m) so that ``H.T @ state`` extracts the observed cells — matching
    ``EnKF::innovation`` (``obs_op.transpose() * ensemble``).
    """
    n_state = truth_state.size
    cell_idx = np.arange(0, n_field, every)
    state_idx = FIELD_OFFSET + cell_idx
    m = cell_idx.size

    H = np.zeros((n_state, m))
    H[state_idx, np.arange(m)] = 1.0
    obs = truth_state[state_idx].copy()  # filter() perturbs observations itself
    return obs, H, float(obs_error), cell_idx, state_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=30, help="ensemble size (Ne)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p_sig", type=float, default=0.05, help="log-normal std for pressures")
    ap.add_argument("--rho_sig", type=float, default=0.10, help="log-normal std for densities")
    ap.add_argument("--obs-every", type=int, default=16,
                    help="observe every k-th density cell (sparse H)")
    ap.add_argument("--obs-error", type=float, default=0.01,
                    help="observation noise std (R = obs_error^2 I)")
    ap.add_argument("--outdir", default=str(_ROOT / "training_data" / "shock_tube" / "enkf_ensemble"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    members_dir = outdir / "members"
    members_dir.mkdir(parents=True, exist_ok=True)

    # --- Truth (unperturbed reference) -----------------------------------
    print(f"[truth] running nominal IC {NOMINAL.tolist()}")
    truth_rho, t_f, x = run_member(NOMINAL, outdir / "truth.h5")
    n_field = truth_rho.size
    truth_state = state_vector(NOMINAL, t_f, truth_rho)
    n_state = truth_state.size
    print(f"[truth] t_f={t_f:.3e}  n_field={n_field}  n_state={n_state}")

    # --- Ensemble members ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    member_params = np.zeros((args.N, N_PARAMS))
    ensemble = np.zeros((n_state, args.N))  # (n_state x Ne): one member per column

    for m in range(args.N):
        params = sample_params(rng, args.p_sig, args.rho_sig)
        rho_final, t_m, _ = run_member(params, members_dir / f"shock_tube__m{m:03d}.h5")
        member_params[m] = params
        ensemble[:, m] = state_vector(params, t_m, rho_final)
        print(f"[m={m:03d}] p_high={params[0]:.4e} p_low={params[1]:.4e} "
              f"rho_high={params[2]:.4f} rho_low={params[3]:.4f}")

    # --- Sparse density observations from truth --------------------------
    obs, H, obs_error, cell_idx, state_idx = build_sparse_obs(
        truth_state, n_field, args.obs_every, args.obs_error
    )
    print(f"[obs] {obs.size} sparse density observations (every {args.obs_every} cells), "
          f"obs_error={obs_error}")

    # --- Save ------------------------------------------------------------
    out_npz = outdir / "enkf_ensemble.npz"
    np.savez(
        out_npz,
        ensemble=ensemble,            # (n_state, Ne)
        truth=truth_state,            # (n_state,)
        member_params=member_params,  # (Ne, 4)
        param_names=np.array(PARAM_NAMES),
        nominal=NOMINAL,
        # state layout
        n_state=n_state,
        n_params=N_PARAMS,            # EnKF "globals" (augmented, inferred)
        time_index=TIME_INDEX,
        field_offset=FIELD_OFFSET,
        n_field=n_field,
        field_key=FIELD_KEY,
        t_f=t_f,
        x=x,
        # observations
        obs=obs,                      # (m,) y^o
        obs_op=H,                     # (n_state, m) H, with H.T @ state -> obs space
        obs_error=obs_error,          # scalar sigma; R = sigma^2 I
        obs_cell_idx=cell_idx,        # observed indices within the 256-cell field
        obs_state_idx=state_idx,      # same indices within the full state vector
        ens_size=args.N,
    )
    print(f"\n[done] ensemble {ensemble.shape}  ->  {out_npz}")
    print(f"       member h5 in {members_dir}/  (also usable as ROSE training data)")


if __name__ == "__main__":
    main()
