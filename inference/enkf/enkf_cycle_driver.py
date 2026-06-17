"""Cycled (time-looping) EnKF for the 1D shock tube.

The cycle version of enkf_driver.py. Instead of a single analysis at the final
time, an ensemble of Euler-solver members is initialised from the 20 HDF5 ICs at
t = 0 and marched to t_final in ``--cycles`` assimilation steps. At each cycle
time t_k:

  1. forecast  -- propagate every member to t_k (ShockTubeMember.forecast);
  2. observe   -- exact Sod density at t_k on the fixed observation cells;
  3. analyse   -- one C++ EnKF step (common.enkf_filter_cpp) on the augmented state;
  4. re-inject -- analysis density + inferred params back into each member,
                  keeping the forecast velocity & pressure, then continue.

Shared infrastructure (ensemble load, exact Sod solution, observation operator,
C++ root filter) comes from ``inference/common.py``; the per-member time
propagation from ``task_simulations/Shock_Tube/propagate.py``. The truth the
observations are drawn from is ``common.TRUTH``.

Run from the repo root::

    python inference/enkf/enkf_cycle_driver.py --cycles 12
    python inference/enkf/enkf_cycle_driver.py --cycles 20 --no-loc
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
    ENSEMBLE_DIR, FIELD_OFFSET, N_FIELD, N_PARAMS, N_STATE, OBS_ERROR,
    T_FINAL, TRUTH,
    enkf_filter_cpp, exact_density_on_cells, load_ensemble, shock_features,
)
from task_simulations.Shock_Tube.propagate import ShockTubeMember  # noqa: E402


def build_obs_operator(t: float, obs_every: int, obs_halfwin: int):
    """Observation operator + cell indices at time ``t``.

    Even baseline (every ``obs_every`` cells) plus a +-``obs_halfwin`` straddle
    at the CURRENT-time Sod features (rarefaction head/tail, contact, shock) --
    so the brackets sit on the discontinuities where they actually are at t,
    not where they end up at t_final. Returns ``(H, cell_idx)``.
    """
    mask = np.zeros(N_FIELD, dtype=bool)
    mask[::obs_every] = True                                       # even baseline
    for fx in shock_features(TRUTH, t):
        c0 = int(np.clip(round(fx * N_FIELD - 0.5), 0, N_FIELD - 1))
        lo, hi = max(0, c0 - obs_halfwin), min(N_FIELD - 1, c0 + obs_halfwin)
        mask[lo:hi + 1] = True                                     # straddle each jump
    cell_idx = np.flatnonzero(mask)
    H = np.zeros((N_STATE, cell_idx.size))
    H[FIELD_OFFSET + cell_idx, np.arange(cell_idx.size)] = 1.0
    return H, cell_idx


def run_cycle(args) -> dict:
    rng = np.random.default_rng(args.seed)

    # --- Initial ensemble: parameters from the 20 HDF5 files, members at t=0 ---
    # We only need the IC rows (the augmented globals), then march each member
    # forward from t = 0.
    ensemble0 = load_ensemble(Path(args.dir))
    params0 = ensemble0[:N_PARAMS, :]                       # (4 x Ne)
    ne = ensemble0.shape[1]
    members = [ShockTubeMember(params0[:, i], nx=N_FIELD) for i in range(ne)]

    localize = not args.no_loc

    # Localization geometry for the STATE rows is time-invariant: globals
    # untapered (placeholder position), local density cells carry their grid index.
    state_loc = np.zeros(N_STATE)
    state_loc[FIELD_OFFSET:] = np.arange(N_FIELD)

    # Evenly spaced assimilation times up to t_final (skip t = 0).
    cycle_times = np.linspace(0.0, args.t_final, args.cycles + 1)[1:]

    # Observation locations: FIXED for every cycle (the SAME cells as the
    # single-step enkf_driver, chosen from the t_final features) or rebuilt each
    # cycle to track the moving discontinuities. With fixed locations only the
    # observed VALUES change with time.
    if args.fixed_obs:
        H_fixed, cell_idx_fixed = build_obs_operator(args.t_final, args.obs_every, args.obs_halfwin)

    print(f"[setup] Ne={ne} cycles={args.cycles} t_final={args.t_final:.2e} "
          f"obs_every={args.obs_every} halfwin={args.obs_halfwin} "
          f"obs_locs={'FIXED@t_final' if args.fixed_obs else 'tracking'} "
          f"reproject={args.reproject} inflation={args.inflation} "
          f"localize={localize} loc_rad={args.loc_rad}")
    print(f"[truth] p_high={TRUTH[0]:.3e} p_low={TRUTH[1]:.3e} "
          f"rho_high={TRUTH[2]:.4f} rho_low={TRUTH[3]:.4f}")

    history = []           # (t, prior_rmse, analysis_rmse)
    spread_history = []    # (t, forecast density spread)
    param_history = []     # (t, mean p_high, p_low, rho_high, rho_low)
    final_members = None   # final-cycle forecast spread, for the grey cloud
    last = {}

    for t_k in cycle_times:
        # 1. Forecast every member to the cycle time.
        for mem in members:
            mem.forecast(t_k)

        # 2. Assemble the forecast ensemble X (n_state x Ne).
        X = np.zeros((N_STATE, ne))
        for i, mem in enumerate(members):
            X[:N_PARAMS, i] = mem.params
            X[N_PARAMS, i] = t_k
            X[FIELD_OFFSET:, i] = mem.density

        # 3. Observation operator (fixed at t_final features, or tracking) and
        #    the observed VALUES drawn from the exact solution at the current time.
        if args.fixed_obs:
            H, cell_idx = H_fixed, cell_idx_fixed
        else:
            H, cell_idx = build_obs_operator(t_k, args.obs_every, args.obs_halfwin)
        obs_loc = cell_idx.astype(float)
        exact_rho = exact_density_on_cells(TRUTH, t_k, N_FIELD)
        y = exact_rho[cell_idx]

        prior_rho = X[FIELD_OFFSET:, :].mean(axis=1)
        prior_rmse = float(np.sqrt(np.mean((prior_rho - exact_rho) ** 2)))

        # 3b. Multiplicative inflation to fight ensemble collapse. Which block
        #     holds the live spread depends on the mode:
        #       - reproject: the field is a deterministic function of the params,
        #         so the PARAMETERS are the degrees of freedom and their cross-cov
        #         with the field is clean -> inflate the params.
        #       - otherwise: inflate the DENSITY field (the globals' problem is the
        #         spurious global pull, not collapse, so leave them alone).
        #     The mean is preserved, so prior_rmse above is unaffected.
        infl = slice(0, N_PARAMS) if args.reproject else slice(FIELD_OFFSET, N_STATE)
        sub = X[infl, :]
        smean = sub.mean(axis=1, keepdims=True)
        fwd_spread = float(X[FIELD_OFFSET:, :].std(axis=1, ddof=1).mean())  # density spread
        if args.inflation != 1.0:
            X[infl, :] = smean + args.inflation * (sub - smean)

        # 4. One C++ EnKF analysis step (root filter in Eigen).
        X_a = enkf_filter_cpp(
            X, y, H, args.obs_error,
            state_loc=state_loc, obs_loc=obs_loc,
            num_globals=FIELD_OFFSET, loc_rad=args.loc_rad, localize=localize,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

        # 5. Update each member: reproject onto the model's solution manifold
        #    (rebuild as the Sod solution for the analyzed params -- sharp, no
        #    overshoot) or, with --no-reproject, re-inject the raw analysis density.
        for i, mem in enumerate(members):
            if args.reproject:
                mem.reproject(X_a[:N_PARAMS, i], t_k)
            else:
                mem.reinject(X_a[FIELD_OFFSET:, i], params=X_a[:N_PARAMS, i])

        # Analysis density that actually carries forward (post-reprojection).
        post_rho = np.column_stack([mem.density for mem in members]).mean(axis=1)
        analysis_rmse = float(np.sqrt(np.mean((post_rho - exact_rho) ** 2)))

        mean_params = np.array([mem.params for mem in members]).mean(axis=0)
        history.append((t_k, prior_rmse, analysis_rmse))
        spread_history.append((t_k, fwd_spread))
        param_history.append((t_k, *mean_params))
        final_members = X[FIELD_OFFSET:, :].copy()         # forecast spread at last t_k
        last = dict(t=t_k, exact=exact_rho, prior=prior_rho, post=post_rho,
                    obs=y, obs_x=cell_idx / N_FIELD)

        print(f"[t={t_k:.2e}] m={cell_idx.size} spread={fwd_spread:.3e} "
              f"prior_rmse={prior_rmse:.4e} analysis_rmse={analysis_rmse:.4e}  "
              f"inferred p_high={mean_params[0]:.3e} rho_high={mean_params[2]:.4f}")

    return {
        "history": np.array(history),
        "spread_history": np.array(spread_history),
        "param_history": np.array(param_history),
        "x_cells": np.arange(N_FIELD) / N_FIELD,           # whole cell points (i/N)
        "final_members": final_members,                    # (N_FIELD x Ne) final-cycle spread
        "init_members": ensemble0[FIELD_OFFSET:, :].copy(),  # initial HDF5 fields (grey cloud)
        "exact_final": last["exact"],
        "prior_final": last["prior"],
        "post_final": last["post"],
        "obs_x": last["obs_x"],                            # final-cycle obs locations
        "obs_y": last["obs"],
        "final_t": last["t"],
    }


def make_plot(res: dict, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = res["x_cells"]
    hist = res["history"]
    fig, (ax_d, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Density at the final cycle: grey ensemble members + analysis mean + obs.
    members = res["final_members"]
    for i in range(members.shape[1]):
        ax_d.plot(x, members[:, i], color="0.8", lw=0.8, zorder=1,
                  label="Ensemble (forecast)" if i == 0 else None)
    ax_d.plot(x, res["post_final"], color="tab:red", lw=1.9, zorder=4,
              label=f"Analysis mean (RMSE {hist[-1, 2]:.2e})")
    ax_d.scatter(res["obs_x"], res["obs_y"], color="gray", marker="o", s=16,
                 edgecolor="white", linewidth=0.4, zorder=5, label="Observations")
    ax_d.set_xlabel("x")
    ax_d.set_ylabel(r"density $\rho$")
    ax_d.set_title(f"Density at final t = {res['final_t']:.2e}")
    ax_d.legend(loc="upper right", framealpha=0.95)
    ax_d.grid(True, alpha=0.3)

    # (b) RMSE across the assimilation cycles (prior vs analysis), with the
    #     forecast ensemble spread on a twin axis to expose collapse.
    ax_r.plot(hist[:, 0], hist[:, 1], "o-", color="tab:blue", label="Prior (forecast)")
    ax_r.plot(hist[:, 0], hist[:, 2], "o-", color="tab:red", label="Analysis")
    ax_r.set_xlabel("t")
    ax_r.set_ylabel("density-field RMSE vs exact")
    ax_r.set_title(f"Assimilation cycles (Ne={members.shape[1]})")
    ax_r.grid(True, alpha=0.3)

    spread = res.get("spread_history")
    if spread is not None and spread.size:
        ax_s = ax_r.twinx()
        ax_s.plot(spread[:, 0], spread[:, 1], color="0.5", lw=1.3, ls=":",
                  label="Forecast spread")
        ax_s.set_ylabel("forecast density spread (std)", color="0.4")
        ax_s.tick_params(axis="y", labelcolor="0.4")
        ax_s.set_ylim(bottom=0.0)
        # Merge legends from both axes.
        h1, l1 = ax_r.get_legend_handles_labels()
        h2, l2 = ax_s.get_legend_handles_labels()
        ax_r.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.95)
    else:
        ax_r.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")


def make_param_plot(res: dict, save_path: Path) -> None:
    """Evolution of the 4 global parameters (ensemble mean) across the cycles,
    normalised to truth so all four share an axis (1.0 = exact recovery)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = res["param_history"]            # (cycles, 5): t, p_high, p_low, rho_high, rho_low
    if ph.size == 0:
        return
    t = ph[:, 0]
    names = ["p_high", "p_low", "rho_high", "rho_low"]
    colors = ["tab:purple", "tab:green", "tab:orange", "tab:brown"]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.axhline(1.0, color="black", lw=0.9, ls=":", label="truth")
    for k, (nm, col) in enumerate(zip(names, colors)):
        ax.plot(t, ph[:, 1 + k] / TRUTH[k], "o-", color=col, lw=1.6, label=nm)
    ax.set_xlabel("t")
    ax.set_ylabel("inferred / truth")
    ax.set_title("Global parameter evolution (multi-step EnKF, ensemble mean)")
    ax.legend(loc="best", framealpha=0.95, ncol=3)
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")


def make_analysis_plot(res: dict, save_path: Path) -> None:
    """enkf_driver.png-style figure (density + residual) using the multi-step
    analysis mean at the final time: exact (black), initial ensemble (grey), the
    multi-step analysis mean (red), observation locations, and a residual panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = res["x_cells"]
    members = res["init_members"]
    exact = res["exact_final"]
    post = res["post_final"]
    prior = members.mean(axis=1)
    post_rmse = res["history"][-1, 2]

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax.plot(x, exact, color="black", lw=2.0, zorder=0, label="Exact")
    for i in range(members.shape[1]):
        ax.plot(x, members[:, i], color="0.8", lw=0.8, zorder=1,
                label="Initial ensemble" if i == 0 else None)
    ax.plot(x, post, color="tab:red", lw=1.9, zorder=4,
            label=f"Multi-step analysis mean (RMSE {post_rmse:.2e})")
    ax.scatter(res["obs_x"], res["obs_y"], color="gray", marker="o", s=16,
               edgecolor="white", linewidth=0.4, zorder=5, label="Observations")
    ax.set_ylabel("density rho")
    ax.set_title(f"Multi-step EnKF analysis  (cycles = {res['param_history'].shape[0]}, "
                 f"t = {res['final_t']:.4f})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    ax_r.axhline(0.0, color="black", lw=0.8)
    ax_r.plot(x, prior - exact, color="tab:blue", lw=1.0, ls="--", label="prior mean")
    ax_r.plot(x, post - exact, color="tab:red", lw=1.4, label="analysis mean")
    ax_r.scatter(res["obs_x"], np.zeros_like(res["obs_x"]), color="gray",
                 marker="|", s=40, zorder=4)
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("residual")
    ax_r.legend(loc="upper right", framealpha=0.95, ncol=2, fontsize=9)
    ax_r.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ENSEMBLE_DIR),
                    help="directory of the 20 ensemble HDF5 files (initial ICs)")
    ap.add_argument("--cycles", type=int, default=12,
                    help="number of assimilation times evenly spaced up to t_final")
    ap.add_argument("--t-final", dest="t_final", type=float, default=T_FINAL,
                    help="final time t_f")
    ap.add_argument("--obs-every", type=int, default=13,
                    help="even baseline: observe every k-th density cell")
    ap.add_argument("--obs-halfwin", dest="obs_halfwin", type=int, default=1,
                    help="cells observed on EACH side of every discontinuity")
    ap.add_argument("--obs-error", type=float, default=OBS_ERROR,
                    help="observation noise std (R = obs_error^2 I)")
    ap.add_argument("--loc-rad", dest="loc_rad", type=float, default=50.0,
                    help="Gaspari-Cohn localization cutoff (cells)")
    ap.add_argument("--no-loc", action="store_true", help="disable localization")
    ap.add_argument("--inflation", type=float, default=1.30,
                    help="multiplicative inflation against ensemble collapse. Applied to "
                         "the params when reprojecting, else to the density field. The "
                         "time-tracking mask diverges without it; pass 1.0 to disable.")
    ap.add_argument("--no-reproject", dest="reproject", action="store_false",
                    help="re-inject the raw analysis density instead of reprojecting each "
                         "member onto the Sod solution manifold for its analyzed params")
    ap.set_defaults(reproject=True)
    ap.add_argument("--fixed-obs", dest="fixed_obs", action="store_true",
                    help="use FIXED observation cells for every cycle (same locations as the "
                         "single-step enkf_driver, from the t_final features) instead of "
                         "rebuilding the mask to track the moving discontinuities")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    args = ap.parse_args()

    res = run_cycle(args)
    if not args.no_plot:
        outdir = Path(args.outdir)
        make_plot(res, outdir / "enkf_cycle_driver.png")
        make_param_plot(res, outdir / "enkf_cycle_params.png")
        make_analysis_plot(res, outdir / "enkf_cycle_analysis.png")


if __name__ == "__main__":
    main()
