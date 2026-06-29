"""Time-cycled EnKF on the 1D shock tube -- 500 assimilation cycles by default.

``run_enkf.py`` does ONE analysis step on a frozen ensemble. This driver instead
marches a live ensemble of Euler-solver members forward in time and assimilates a
density observation at every cycle, so the 4 Sod ICs are refined over many
updates. Per cycle:

  1. forecast  -- advance every member to the next cycle time with EulerSolver1D;
  2. observe   -- exact Sod density at the truth operating point, sampled in the
                  region interiors at the CURRENT time (features move, so the obs
                  mask is rebuilt each cycle);
  3. analyse   -- one C++ stochastic EnKF step on the augmented state
                  [ p_high, p_low, rho_high, rho_low, t,  rho_0 ... rho_255 ];
  4. update    -- carry the analysis ICs forward; the density field is NOT
                  injected back. Each cycle re-solves every member from its
                  current ICs, so the forecast field is always a valid Euler
                  solution -- this is the parameter-space cycled EnKF, and it
                  avoids the divergence that injecting an inconsistent
                  (analysis-density / forecast-velocity) state causes over many
                  cycles.

Final time
----------
The single-shot forward model stops at T_FINAL = 6e-4. 500 cycles need a longer
horizon, so ``--t-final`` defaults to 9e-4 -- still inside the window where every
Sod wave (rarefaction head .. shock) sits in [0, 1], so the analytic observations
stay valid. dt_cycle = t_final / n_cycles.

A modest multiplicative covariance inflation (``--inflation``, default 1.02)
counteracts the ensemble collapse that 500 repeated updates would otherwise cause,
keeping the parameter posterior visible.

Run from the repo root::

    python EnKF/run_enkf_cycle.py
    python EnKF/run_enkf_cycle.py --cycles 500 --N 30 --t-final 9e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# This script lives one level under the repo root (EnKF/).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from EnKF.ensemble_common import (  # noqa: E402
    FIELD_OFFSET, N_FIELD, N_PARAMS, N_STATE, OBS_ERROR, PARAM_NAMES, TRUTH,
    build_observations, enkf_filter_cpp, record_parameters,
)
from task_simulations.Shock_Tube.sod_euler import EulerSolver1D  # noqa: E402

_HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = _HERE / "results"

# Box constraint for the ICs (a prior). The augmented-state EnKF's global-pull
# can drag a parameter past zero (NaN'ing the Euler solve and poisoning the whole
# ensemble covariance) or to an extreme where the sound speed explodes and the
# CFL substepping hangs. Clamping each member into [lo, hi] = [0.2, 5] x truth
# keeps every member physical, solvable, and fast.
CLIP_LO_FRAC = 0.2
CLIP_HI_FRAC = 5.0


def run(args) -> dict:
    rng = np.random.default_rng(args.seed)
    truth = TRUTH if args.truth is None else np.asarray(args.truth, dtype=float)
    localize = not args.no_loc
    clip_lo, clip_hi = CLIP_LO_FRAC * truth, CLIP_HI_FRAC * truth

    def euler_field(params, t):
        """Density field of a fresh Euler solve from `params` to time t."""
        solver = EulerSolver1D(nx=N_FIELD, xmin=0.0, xmax=1.0, gamma=1.4, cfl=0.5)
        solver.set_sod_like(rho_high=params[2], p_high=params[0],
                            rho_low=params[3], p_low=params[1], x0=0.5)
        solver.step_to(t)
        return solver.U[0]

    # --- ensemble carried in PARAMETER space: perturbed ICs ------------------
    member_params = np.zeros((args.N, N_PARAMS))
    for i in range(args.N):
        member_params[i] = truth * (1.0 + args.gen_spread * rng.standard_normal(N_PARAMS))

    dt = args.t_final / args.cycles
    print(f"[setup] Ne={args.N} cycles={args.cycles} t_final={args.t_final:.3e} "
          f"dt={dt:.3e} inflation={args.inflation} localize={localize} "
          f"loc_rad={args.loc_rad}")
    print(f"[truth] p_high={truth[0]:.3e} p_low={truth[1]:.3e} "
          f"rho_high={truth[2]:.4f} rho_low={truth[3]:.4f}")

    # State-space localization geometry (globals untapered, density cells indexed).
    state_loc = np.zeros(N_STATE)
    state_loc[FIELD_OFFSET:] = np.arange(N_FIELD)

    prior_params0 = member_params.mean(axis=0).copy()
    param_hist = []   # (t, p_high, p_low, rho_high, rho_low) analysis means
    std_hist = []     # (t, std of each param)
    rmse_hist = []    # (t, prior_rmse, analysis_rmse)
    exact_final = prior_rho = post_rho = obs_x = obs_y = None
    final_t = 0.0

    for k in range(1, args.cycles + 1):
        t_k = k * dt

        # 1. forecast: re-solve every member from its CURRENT ICs to t_k
        fields = np.array([euler_field(member_params[i], t_k) for i in range(args.N)])  # (N x N_FIELD)

        # 2. observations: exact Sod density in the region interiors at t_k
        H, obs, cell_idx, exact_rho = build_observations(
            args.obs_every, args.margin, truth=truth, t=t_k,
            obs_error=args.obs_error, seed=int(rng.integers(0, 2**31 - 1)))
        obs_loc = cell_idx.astype(float)

        # 3. assemble forecast ensemble, with multiplicative inflation of anomalies
        X = np.zeros((N_STATE, args.N))
        X[:N_PARAMS, :] = member_params.T
        X[N_PARAMS, :] = t_k
        X[FIELD_OFFSET:, :] = fields.T
        if args.inflation != 1.0:
            xbar = X.mean(axis=1, keepdims=True)
            X = xbar + args.inflation * (X - xbar)

        prior_rho = X[FIELD_OFFSET:, :].mean(axis=1)
        prior_rmse = float(np.sqrt(np.mean((prior_rho - exact_rho) ** 2)))

        # 4. one C++ EnKF analysis step
        X_a = enkf_filter_cpp(
            X, obs, H, args.obs_error,
            state_loc=state_loc, obs_loc=obs_loc,
            num_globals=FIELD_OFFSET, loc_rad=args.loc_rad, localize=localize,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        post_rho = X_a[FIELD_OFFSET:, :].mean(axis=1)
        analysis_rmse = float(np.sqrt(np.mean((post_rho - exact_rho) ** 2)))

        # 5. update ICs only -- next cycle re-solves the field from them.
        #    Box-clip so an over-pulled global can't NaN/stall the solver.
        member_params = np.clip(X_a[:N_PARAMS, :].T, clip_lo, clip_hi)

        pmean = member_params.mean(axis=0)
        param_hist.append((t_k, *pmean))
        std_hist.append((t_k, *member_params.std(axis=0)))
        rmse_hist.append((t_k, prior_rmse, analysis_rmse))
        exact_final, obs_x, obs_y, final_t = exact_rho, cell_idx / N_FIELD, obs, t_k

        if k % max(1, args.cycles // 20) == 0 or k == 1:
            print(f"[cycle {k:4d} t={t_k:.2e}] prior_rmse={prior_rmse:.3e} "
                  f"analysis_rmse={analysis_rmse:.3e}  "
                  f"p_high={pmean[0]:.3e} rho_high={pmean[2]:.4f}")

    final_params = member_params.copy()           # (N x 4) analysis ensemble
    pmean = final_params.mean(axis=0)
    print("\n[result] final analysis parameter means (vs truth):")
    for k, nm in enumerate(PARAM_NAMES):
        rel = abs(pmean[k] - truth[k]) / abs(truth[k])
        print(f"  {nm:9s} {pmean[k]:.4e}  (truth {truth[k]:.4e}, rel err {rel:.2%})")

    # State vector reflecting the FINAL inferred ICs (re-solve each member to t_final).
    post_members_rho = np.array([euler_field(final_params[i], final_t)
                                 for i in range(args.N)]).T   # (N_FIELD x N)
    post_mean_rho = post_members_rho.mean(axis=1)
    exact_final = build_observations(args.obs_every, args.margin, truth=truth,
                                     t=final_t, obs_error=0.0)[3]

    return {
        "final_params": final_params,                 # (N x 4)
        "param_mean": pmean,
        "prior_params0": prior_params0,
        "truth": truth,
        "param_hist": np.array(param_hist),
        "std_hist": np.array(std_hist),
        "rmse_hist": np.array(rmse_hist),
        "post_members_rho": post_members_rho,         # (N_FIELD x N)
        "post_mean_rho": post_mean_rho,
        "exact_final": exact_final,
        "obs_x": obs_x, "obs_y": obs_y, "final_t": final_t,
    }


# --------------------------------------------------------------------------- #
# Figure: 4 parameter posterior histograms (mean + truth lines) and the mean
# analysis state vector vs the exact field.
# --------------------------------------------------------------------------- #
def make_plot(res: dict, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = res["final_params"]      # (N x 4)
    truth = res["truth"]
    pmean = res["param_mean"]

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])

    # --- left 2x2 block: parameter marginals -------------------------------
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for k, (r, c) in enumerate(cells):
        ax = fig.add_subplot(gs[r, c])
        ax.hist(params[:, k], bins=min(25, max(8, params.shape[0] // 2)),
                color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(pmean[k], color="tab:green", lw=2.0,
                   label=f"mean = {pmean[k]:.4g}")
        ax.axvline(truth[k], color="red", lw=2.0, ls="--",
                   label=f"truth = {truth[k]:.4g}")
        ax.set_title(PARAM_NAMES[k])
        ax.set_xlabel(PARAM_NAMES[k]); ax.set_ylabel("count")
        ax.legend(fontsize=8)

    # --- right column (spans both rows): mean state vector -----------------
    ax_s = fig.add_subplot(gs[:, 2])
    x = np.arange(N_FIELD) / N_FIELD
    members_rho = res["post_members_rho"]
    band_lo = members_rho.mean(axis=1) - members_rho.std(axis=1)
    band_hi = members_rho.mean(axis=1) + members_rho.std(axis=1)
    rmse = float(np.sqrt(np.mean((res["post_mean_rho"] - res["exact_final"]) ** 2)))

    ax_s.plot(x, res["exact_final"], color="black", lw=2.2, label="Exact (Sod)")
    ax_s.fill_between(x, band_lo, band_hi, color="tab:red", alpha=0.2,
                      label="analysis ±1σ")
    ax_s.plot(x, res["post_mean_rho"], color="tab:red", lw=1.8,
              label=f"Analysis mean (RMSE {rmse:.2e})")
    ax_s.scatter(res["obs_x"], res["obs_y"], color="gray", marker="o", s=14,
                 edgecolor="white", linewidth=0.4, zorder=5, label="Observations")
    ax_s.set_xlabel("x"); ax_s.set_ylabel(r"density $\rho$")
    ax_s.set_title(f"Mean state vector at final t = {res['final_t']:.2e}")
    ax_s.legend(loc="upper right", framealpha=0.95)
    ax_s.grid(True, alpha=0.3)

    fig.suptitle(f"Cycled EnKF ({res['param_hist'].shape[0]} assimilation cycles): "
                 "parameter posteriors + mean state", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=500, help="number of assimilation cycles")
    ap.add_argument("--N", type=int, default=30, help="ensemble size (Ne)")
    ap.add_argument("--t-final", dest="t_final", type=float, default=9.0e-4,
                    help="final time (extended from 6e-4 to fit many cycles; keep <~9e-4 "
                         "so every Sod wave stays inside [0,1])")
    ap.add_argument("--inflation", type=float, default=1.02,
                    help="multiplicative covariance inflation per cycle (1.0 = off)")
    ap.add_argument("--gen-spread", dest="gen_spread", type=float, default=0.1,
                    help="relative IC std of the initial ensemble")
    ap.add_argument("--truth", type=float, nargs=4, default=None,
                    metavar=("P_HIGH", "P_LOW", "RHO_HIGH", "RHO_LOW"))
    ap.add_argument("--obs-every", type=int, default=15)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--obs-error", dest="obs_error", type=float, default=OBS_ERROR)
    ap.add_argument("--loc-rad", dest="loc_rad", type=float, default=50.0)
    ap.add_argument("--no-loc", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--no-plot", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    res = run(args)
    record_parameters(
        np.vstack([res["prior_params0"][:, None].repeat(res["final_params"].shape[0], 1)]),
        res["final_params"].T, Path(args.outdir) / "run_enkf_cycle_params.json",
        truth=res["truth"])
    if not args.no_plot:
        make_plot(res, Path(args.outdir) / "run_enkf_cycle.png")


if __name__ == "__main__":
    main()
