"""Ensemble Smoother with Multiple Data Assimilation (ES-MDA) for the 1D shock tube.

Entirely Python (no C++), as a counterpart to the single-step EnKF in
enkf_driver.py. Where enkf_driver does ONE augmented-state Kalman update, ES-MDA
(Emerick & Reynolds, 2013) performs ``Na`` tempered assimilations of the SAME
observations, re-running the forward model between iterations -- which handles
the nonlinear parameters->density map far better than a single linear update.

Formulation here is PARAMETER estimation:
  - ensemble state  m = [p_high, p_low, rho_high, rho_low]  (the 4 ICs)
  - forward model   g(m) = density at the observed cells at T_FINAL, from
    ``--forward`` (exact | euler | surrogate; see ``common.make_forward``)
  - data            d = exact Sod density at those cells for the TRUTH params
    (SAME obs methodology as enkf_driver / the samplers)

Each iteration i = 1..Na (inflation alpha_i, with sum_i 1/alpha_i = 1):
  D_i      = g(M_i)                              # predicted data per member (m x Ne)
  d_uc,j   = d + sqrt(alpha_i) * sqrt(C_D) z_j   # perturbed obs, z ~ N(0, I)
  C_MD     = cov(M_i, D_i)                       # state-data cross-cov (P x m)
  C_DD     = cov(D_i)                            # data auto-cov (m x m)
  M_{i+1}  = M_i + C_MD (C_DD + alpha_i C_D)^{-1} (d_uc - D_i)

Because the analysis state is the parameters, the reconstructed field g(m) is
always a valid, sharp Sod profile -- so there is no at-jump overshoot by
construction (unlike the augmented-state EnKF).

Run from the repo root::

    python inference/enkf/es_mda.py
    python inference/enkf/es_mda.py --na 8 --no-compare
    python inference/enkf/es_mda.py --forward surrogate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# This script lives two levels under the repo root (inference/enkf/).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.common import (  # noqa: E402
    ENSEMBLE_DIR, FIELD_OFFSET, FORWARD_MODES, N_FIELD, N_PARAMS, OBS_ERROR,
    PARAM_NAMES, T_FINAL, TRUTH,
    build_observations, load_ensemble, make_forward, plot_field_comparison,
)


def _field_ensemble(M: np.ndarray, field_fn) -> np.ndarray:
    """Forward-model density field for each member's params -> (N_FIELD x Ne)."""
    return np.column_stack([field_fn(M[:, j]) for j in range(M.shape[1])])


def run_es_mda(args) -> dict:
    rng = np.random.default_rng(args.seed)
    field_fn = make_forward(args.forward)            # full N_FIELD field

    # Prior parameter ensemble = the 4 IC rows of the 20 HDF5 members.
    ensemble0 = load_ensemble(Path(args.dir))
    M = ensemble0[:N_PARAMS, :].copy()                 # (4 x Ne)
    ne = M.shape[1]
    prior_field_members = ensemble0[FIELD_OFFSET:, :]  # HDF5 fields, for the grey cloud

    # Observation methodology (identical to enkf_driver / the samplers).
    H, d_obs, cell_idx, exact_rho = build_observations(args.obs_every, args.margin)
    m = d_obs.size
    cd = args.obs_error ** 2                            # obs variance (C_D = cd * I)

    # Constant inflation schedule: alpha_i = Na, so sum_i 1/alpha_i = 1.
    na = args.na
    alphas = np.full(na, float(na))

    prior_mean_field = prior_field_members.mean(axis=1)
    prior_rmse = float(np.sqrt(np.mean((prior_mean_field - exact_rho) ** 2)))
    print(f"[setup] ES-MDA Ne={ne} m={m} Na={na} forward={args.forward} "
          f"obs_every={args.obs_every} margin={args.margin} obs_error={args.obs_error}")
    print(f"[truth] p_high={TRUTH[0]:.3e} p_low={TRUTH[1]:.3e} "
          f"rho_high={TRUTH[2]:.4f} rho_low={TRUTH[3]:.4f}")
    print(f"[prior] density RMSE vs exact = {prior_rmse:.4e}")

    # Forward-model field for the current ensemble (reused as the next iteration's
    # predicted data -> Na+1 evals total, which matters when g is the Euler solver).
    field = _field_ensemble(M, field_fn)               # (N_FIELD x Ne)

    history = []                                        # (it, rmse, p_high, p_low, rho_high, rho_low)
    for it, alpha in enumerate(alphas, 1):
        D = field[cell_idx, :]                          # predicted data g(M) at obs cells (m x Ne)
        # Perturbed observations, inflated by sqrt(alpha).
        D_uc = d_obs[:, None] + np.sqrt(alpha) * args.obs_error * rng.standard_normal((m, ne))

        # Ensemble covariances.
        dM = M - M.mean(axis=1, keepdims=True)
        dD = D - D.mean(axis=1, keepdims=True)
        C_MD = dM @ dD.T / (ne - 1)                     # (4 x m)
        C_DD = dD @ dD.T / (ne - 1)                     # (m x m)

        # Tempered Kalman-like update of the parameters.
        S = C_DD + alpha * cd * np.eye(m)
        K = C_MD @ np.linalg.pinv(S)                    # (4 x m)
        M = M + K @ (D_uc - D)
        M = np.maximum(M, 1e-6)                         # keep params positive

        field = _field_ensemble(M, field_fn)            # post-update field (reused next iter)
        rmse = float(np.sqrt(np.mean((field.mean(axis=1) - exact_rho) ** 2)))
        mp = M.mean(axis=1)
        history.append((it, rmse, *mp))
        print(f"[it {it}/{na} alpha={alpha:.1f}] rmse={rmse:.4e}  "
              f"p_high={mp[0]:.4e} p_low={mp[1]:.4e} "
              f"rho_high={mp[2]:.4f} rho_low={mp[3]:.4f}")

    post_field_members = field
    post_mean = post_field_members.mean(axis=1)
    post_rmse = float(np.sqrt(np.mean((post_mean - exact_rho) ** 2)))
    params = M.mean(axis=1)

    # Fair metric: RMSE against g(TRUTH) -- the field the TRUE params produce in
    # the SAME forward model -- so the model's own error (shock smearing) is
    # removed and only ES-MDA's parameter recovery is measured. For
    # forward="exact" this equals post_rmse (g(TRUTH) == exact).
    truth_field = field_fn(TRUTH)
    model_rmse = float(np.sqrt(np.mean((post_mean - truth_field) ** 2)))
    print(f"[result] RMSE vs exact = {post_rmse:.4e}   "
          f"RMSE vs g(truth) = {model_rmse:.4e}  (isolates param recovery)")

    return {
        "exact": exact_rho,
        "prior_members": prior_field_members,
        "post_mean": post_mean,
        "obs_x": cell_idx / N_FIELD,
        "obs_y": d_obs,
        "prior_rmse": prior_rmse,
        "post_rmse": post_rmse,
        "params": params,
        "history": np.array(history),
        "na": na,
        "forward": args.forward,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ENSEMBLE_DIR),
                    help="directory of the 20 ensemble HDF5 files (prior)")
    ap.add_argument("--na", type=int, default=4, help="number of assimilations (ES-MDA iterations)")
    ap.add_argument("--forward", choices=FORWARD_MODES, default="exact",
                    help="forward model g(m): 'exact' analytic Sod (fast, but same model as "
                         "the obs -> inverse crime), 'euler' numerical solver (honest, "
                         "slower), or 'surrogate' trained GPR")
    ap.add_argument("--obs-every", type=int, default=15,
                    help="baseline: observe every k-th density cell in the flat regions")
    ap.add_argument("--margin", type=int, default=4,
                    help="exclude cells within this many of each discontinuity "
                         "(observe only the 5 region interiors)")
    ap.add_argument("--obs-error", type=float, default=OBS_ERROR,
                    help="observation noise std (C_D = obs_error^2 I)")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the single-step EnKF baseline comparison")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = ap.parse_args()

    res = run_es_mda(args)
    if not args.no_plot:
        plot_field_comparison(
            exact=res["exact"], members=res["prior_members"],
            post_mean=res["post_mean"], obs_x=res["obs_x"], obs_y=res["obs_y"],
            post_rmse=res["post_rmse"],
            title=f"ES-MDA  (Na = {res['na']},  g = {res['forward']},  t = {T_FINAL:.4f})",
            post_label="ES-MDA analysis mean", prior_label="Initial ensemble",
            save_path=Path(args.outdir) / "es_mda.png",
        )

    # --- direct comparison against the single-step EnKF -----------------
    if not args.no_compare:
        from inference.enkf import enkf_driver
        print("\n--- single-step EnKF baseline (same observations) ---")
        enkf_args = enkf_driver.build_parser().parse_args([
            "--obs-every", str(args.obs_every), "--margin", str(args.margin),
            "--obs-error", str(args.obs_error), "--seed", str(args.seed),
            "--no-plot",
        ])
        enkf_res = enkf_driver.run(enkf_args)
        print("\n=== comparison (density RMSE vs exact) ===")
        print(f"  prior            : {res['prior_rmse']:.4e}")
        print(f"  single-step EnKF : {enkf_res['post_rmse']:.4e}")
        print(f"  ES-MDA (Na={res['na']})    : {res['post_rmse']:.4e}")
        print("=== ES-MDA inferred params vs truth ===")
        for k, nm in enumerate(PARAM_NAMES):
            print(f"  {nm:9s} {res['params'][k]:.4e}   (truth {TRUTH[k]:.4e})")


if __name__ == "__main__":
    main()
