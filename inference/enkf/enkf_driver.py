"""Single-step EnKF for the 1D shock tube.

The forecast (prior ensemble) is produced ON THE FLY by a forward model
(``--forward``, default the numerical Euler solver). Each ensemble member draws
its 4 shock-tube ICs from a Gaussian around a prior mean and is pushed to the
final time; the resulting density field is the member's local state. The pieces
(all shared infrastructure lives in ``inference/common.py``):

  1. FORECAST -- ``common.forecast_ensemble``: draw Ne perturbed IC sets,
     forward-solve each to T_FINAL, and stack every run into a 261-entry
     augmented state vector
         [ p_high, p_low, rho_high, rho_low, t,  rho_0 ... rho_255 ]
          |<------------ 5 globals ----------->|  |<-- 256-cell density -->|
  2. OBSERVE -- ``common.build_observations``: exact Sod density at the final
     time, placed in the interiors of the 5 Sod regions (never on a front);
  3. ANALYZE -- one C++ EnKF analysis step (EnKF/enkf_step, built from the
     reference EnKF.h) at the final time;
  4. PLOT -- observation locations, analysis ensemble mean, and every forecast
     member in light grey.

The truth operating point the observations are drawn from is ``common.TRUTH``.

Run from the repo root::

    python inference/enkf/enkf_driver.py
    python inference/enkf/enkf_driver.py --ne 40 --no-loc --loc-rad 50
    python inference/enkf/enkf_driver.py --forward surrogate
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
    FIELD_OFFSET, FORWARD_MODES, N_FIELD, N_PARAMS, OBS_ERROR, PARAM_NAMES,
    PRIOR_SPREAD, T_FINAL, TRUTH,
    build_observations, enkf_filter_cpp, forecast_ensemble, make_forward,
    plot_field_comparison,
)


def run(args) -> dict:
    rng = np.random.default_rng(args.seed)
    spread = PRIOR_SPREAD if args.prior_spread is None else np.full(N_PARAMS, args.prior_spread)
    ensemble, _ = forecast_ensemble(rng, args.ne, prior_spread=spread, forward=args.forward)
    ne = ensemble.shape[1]
    H, obs, cell_idx, exact_rho = build_observations(args.obs_every, args.margin)
    localize = not args.no_loc

    # Localization geometry: globals carry a placeholder position (left
    # untapered); local density cells carry their grid index.
    state_loc = np.zeros(ensemble.shape[0])
    state_loc[FIELD_OFFSET:] = np.arange(N_FIELD)
    obs_loc = cell_idx.astype(float)

    print(f"[setup] Ne={ne} m={cell_idx.size} forward={args.forward} "
          f"obs_every={args.obs_every} margin={args.margin} (region interiors only) "
          f"localize={localize} loc_rad={args.loc_rad}")
    print(f"[truth] p_high={TRUTH[0]:.3e} p_low={TRUTH[1]:.3e} "
          f"rho_high={TRUTH[2]:.4f} rho_low={TRUTH[3]:.4f} t={T_FINAL:.2e}")

    # --- One C++ EnKF analysis step ------------------------------------
    X_a = enkf_filter_cpp(
        ensemble, obs, H, args.obs_error,
        state_loc=state_loc, obs_loc=obs_loc,
        num_globals=FIELD_OFFSET, loc_rad=args.loc_rad, localize=localize,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    prior_mean = ensemble.mean(axis=1)
    post_mean = X_a.mean(axis=1)
    prior_rmse = float(np.sqrt(np.mean((prior_mean[FIELD_OFFSET:] - exact_rho) ** 2)))
    post_rmse = float(np.sqrt(np.mean((post_mean[FIELD_OFFSET:] - exact_rho) ** 2)))

    print(f"[rmse] prior={prior_rmse:.4e}  analysis={post_rmse:.4e}")
    print("[params] inferred global mean (prior -> analysis vs truth):")
    for k, nm in enumerate(PARAM_NAMES):
        print(f"  {nm:9s} {prior_mean[k]:.4e} -> {post_mean[k]:.4e}   "
              f"(truth {TRUTH[k]:.4e})")

    return {
        "exact": exact_rho,
        "prior_members": ensemble[FIELD_OFFSET:, :],    # (N_FIELD x Ne) forecast field
        "post_mean": post_mean[FIELD_OFFSET:],
        "obs_x": cell_idx / N_FIELD,
        "obs_y": obs,
        "prior_rmse": prior_rmse,
        "post_rmse": post_rmse,
        "forward": args.forward,
    }


def verify_perturbation(args) -> None:
    """Confirm the C++ filter perturbs observations independently per member.

    Asks the filter to also return its perturbed observation ensemble y^o_e
    (m x Ne) and checks: (1) every member column is distinct, (2) the
    across-member mean of each observation row is ~ y^o (zero-mean noise),
    (3) the across-member std is ~ obs_error, and (4) the perturbation is
    controlled by the seed (same seed reproduces, different seed changes).
    """
    rng = np.random.default_rng(args.seed)
    spread = PRIOR_SPREAD if args.prior_spread is None else np.full(N_PARAMS, args.prior_spread)
    ensemble, _ = forecast_ensemble(rng, args.ne, prior_spread=spread, forward=args.forward)
    ne = ensemble.shape[1]
    H, obs, cell_idx, _ = build_observations(args.obs_every, args.margin)
    m = obs.size
    localize = not args.no_loc
    state_loc = np.zeros(ensemble.shape[0])
    state_loc[FIELD_OFFSET:] = np.arange(N_FIELD)
    obs_loc = cell_idx.astype(float)

    def filt(seed):
        return enkf_filter_cpp(
            ensemble, obs, H, args.obs_error,
            state_loc=state_loc, obs_loc=obs_loc,
            num_globals=FIELD_OFFSET, loc_rad=args.loc_rad, localize=localize,
            seed=seed, return_perturbed=True,
        )

    _, P = filt(111)            # perturbed observations y^o_e (m x Ne)
    _, P_same = filt(111)       # same seed -> identical perturbation
    _, P_diff = filt(222)       # different seed -> different perturbation

    cols_distinct = (np.unique(P, axis=1).shape[1] == ne)
    row_mean = P.mean(axis=1)
    row_std = P.std(axis=1, ddof=1)

    print(f"\n[verify] perturbed observations y^o_e: shape (m={m} x Ne={ne})")
    print(f"  every member column distinct        : {cols_distinct}")
    print(f"  max |across-member mean - y^o|      : {np.max(np.abs(row_mean - obs)):.3e}"
          f"   (zero-mean noise -> small)")
    print(f"  mean across-member std              : {row_std.mean():.3e}"
          f"   (target sigma = {args.obs_error:.2e})")
    print(f"  obs row 0: y^o={obs[0]:.5f}  member draws[:6]="
          f"{np.array2string(P[0, :6], precision=5, floatmode='fixed')}")
    print(f"[verify] same seed reproduces y^o_e   : {np.allclose(P, P_same)}")
    print(f"[verify] different seed changes y^o_e : {not np.allclose(P, P_diff)}")


def make_forward_comparison(res: dict, save_path: Path) -> None:
    """Compare three states at the final time: the EnKF analysis, the forecast
    model run with the SAME parameters the observations were generated from
    (TRUTH), and the analytic (exact) solution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fwd_rho = make_forward(res["forward"])(TRUTH)

    cells = np.arange(N_FIELD)            # grid cell number on the x-axis
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    ax.plot(cells, res["exact"], color="black", lw=2.0, zorder=1, label="Analytical")
    ax.plot(cells, res["post_mean"], color="tab:blue", lw=1.8, zorder=2, label="Analysis")
    ax.plot(cells, fwd_rho, color="red", lw=1.8, ls=":", zorder=3,
            label=f"Forward model ({res['forward']}) at truth")
    ax.set_xlabel("grid cell number")
    ax.set_ylabel("density rho")
    ax.set_title(f"State comparison  (t = {T_FINAL:.4f})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ne", type=int, default=20,
                    help="forecast ensemble size (members forward-solved)")
    ap.add_argument("--forward", choices=FORWARD_MODES, default="euler",
                    help="forecast model the prior members are pushed with")
    ap.add_argument("--prior-spread", dest="prior_spread", type=float, default=None,
                    help="relative IC std for the forecast prior (single value applied "
                         "to all 4 params; default: per-param PRIOR_SPREAD)")
    ap.add_argument("--obs-every", type=int, default=15,
                    help="baseline: observe every k-th density cell in the flat regions")
    ap.add_argument("--margin", type=int, default=4,
                    help="exclude cells within this many of each discontinuity, so "
                         "observations land only in the 5 region interiors")
    ap.add_argument("--obs-error", type=float, default=OBS_ERROR,
                    help="observation noise std (R = obs_error^2 I)")
    ap.add_argument("--loc-rad", dest="loc_rad", type=float, default=50.0,
                    help="Gaspari-Cohn localization cutoff (cells)")
    ap.add_argument("--no-loc", action="store_true", help="disable localization")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    ap.add_argument("--verify-perturb", dest="verify_perturb", action="store_true",
                    help="also confirm the filter perturbs observations per member")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    res = run(args)
    if not args.no_plot:
        plot_field_comparison(
            exact=res["exact"], members=res["prior_members"],
            post_mean=res["post_mean"], obs_x=res["obs_x"], obs_y=res["obs_y"],
            post_rmse=res["post_rmse"],
            title=f"Single-step EnKF, {res['forward']} forecast  (t = {T_FINAL:.4f})",
            post_label="Analysis mean", prior_label="Forecast ensemble",
            save_path=Path(args.outdir) / "enkf_driver.png",
        )
        make_forward_comparison(res, Path(args.outdir) / "enkf_driver_forward_compare.png")
    if args.verify_perturb:
        verify_perturbation(args)


if __name__ == "__main__":
    main()
