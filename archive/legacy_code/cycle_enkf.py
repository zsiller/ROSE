"""Cycled (time-looping) EnKF for the 1D shock tube.

An ensemble of Euler-solver members is marched in time from t_i = 0 to
t_f = 6.0e-4. Every ``--frequency`` micro-steps the filter assimilates density
observations drawn from the EXACT Sod solution at the current time, then injects
the analysis back into each member and keeps forecasting.

Per cycle:
  1. forecast  -- advance every member to the next time with EulerSolver1D;
  2. observe   -- exact_density_on_cells(truth, t) at the fixed observation cells;
  3. analyse   -- one stochastic EnKF step (enkf_analysis), density-only state
                  augmented with the 4 ICs + t (globals) for parameter tracking;
  4. re-inject -- set each member's density to the analysis, keep the forecast
                  velocity & pressure (set_state_primitive), then continue.

State vector per member (column), length n_state = 5 + n_field:
  [ p_high, p_low, rho_high, rho_low, t,  rho_0 ... rho_{n_field-1} ]
   |<------------ 5 globals ----------->|  |<------ density field ------>|

Run from the repo root::

    python task_simulations/Shock_Tube/cycle_enkf.py --N 20 --frequency 10
    python task_simulations/Shock_Tube/cycle_enkf.py --frequency 5 --no-loc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# This script lives two levels under the repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task_simulations.Shock_Tube.sod_euler import EulerSolver1D
from task_simulations.Shock_Tube.sod_exact import exact_density_on_cells, shock_features
from task_simulations.Shock_Tube.enkf_analysis import enkf_analysis
from task_simulations.Shock_Tube.enkf_cpp import enkf_filter_cpp

# Canonical truth operating point the observations are drawn from.
NOMINAL = np.array([1.0e5, 1.0e4, 1.0, 0.125])  # p_high, p_low, rho_high, rho_low
N_PARAMS = 4
FIELD_OFFSET = N_PARAMS + 1  # density field starts at row 5 (after 4 ICs + t)


def sample_params(rng: np.random.Generator, p_sig: float, rho_sig: float) -> np.ndarray:
    """Log-normal multiplicative perturbation of the four ICs (cf. gen_enkf_ensemble)."""
    z = rng.standard_normal(N_PARAMS)
    sig = np.array([p_sig, p_sig, rho_sig, rho_sig])
    return NOMINAL * np.exp(sig * z)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_cycle_enkf(args) -> dict:
    rng = np.random.default_rng(args.seed)
    nx = args.nx

    # --- Initialise ensemble members -----------------------------------
    members: list[EulerSolver1D] = []
    member_params = np.zeros((args.N, N_PARAMS))
    for i in range(args.N):
        params = sample_params(rng, args.p_sig, args.rho_sig)
        solver = EulerSolver1D(nx=nx, xmin=0.0, xmax=1.0, gamma=1.4, cfl=0.5)
        solver.set_sod_like(rho_high=params[2], p_high=params[0],
                            rho_low=params[3], p_low=params[1], x0=args.x0)
        members.append(solver)
        member_params[i] = params

    # --- Fixed observation operator over the density cells -------------
    # Locations are chosen ONCE from the exact solution's discontinuity layout
    # at the FINAL time, then reused unchanged every cycle: an even baseline
    # (every args.obs_every cells, so the plateaus and the rarefaction slope are
    # all sampled) plus a +-args.obs_halfwin straddle bracketing each feature
    # (rarefaction head/tail, contact, shock) to catch the top and bottom state
    # across every jump.
    n_field = nx
    n_state = FIELD_OFFSET + n_field

    mask = np.zeros(n_field, dtype=bool)
    mask[::args.obs_every] = True                              # even baseline coverage
    for fx in shock_features(NOMINAL, args.t_final, x0=args.x0):
        c0 = int(np.clip(round(fx * n_field - 0.5), 0, n_field - 1))
        lo, hi = max(0, c0 - args.obs_halfwin), min(n_field - 1, c0 + args.obs_halfwin)
        mask[lo:hi + 1] = True                                # straddle each discontinuity
    cell_idx = np.flatnonzero(mask)                           # same locations every cycle
    state_idx = FIELD_OFFSET + cell_idx
    m = cell_idx.size
    H = np.zeros((n_state, m))
    H[state_idx, np.arange(m)] = 1.0

    # Localization geometry: globals carry a placeholder position (left
    # untapered inside enkf_analysis); local cells carry their grid index.
    state_loc = np.zeros(n_state)
    state_loc[FIELD_OFFSET:] = np.arange(n_field)
    obs_loc = cell_idx.astype(float)

    x_cells = np.arange(n_field) / n_field   # whole cell points (i/N), not centers
    n_steps = int(round(args.t_final / args.dt))
    localize = not args.no_loc

    print(f"[setup] Ne={args.N} nx={nx} obs_every={args.obs_every} "
          f"halfwin={args.obs_halfwin} (m={m}, fixed at t_final) "
          f"frequency={args.frequency} steps={n_steps} "
          f"localize={localize} loc_rad={args.loc_rad} filter={args.filter}")

    # --- Time loop ------------------------------------------------------
    history = []           # (t, prior_rmse, analysis_rmse)
    param_history = []     # (t, inferred params mean)
    last_prior_rho = None
    last_post_rho = None
    last_obs = None
    last_t = 0.0

    for step in range(1, n_steps + 1):
        t_target = step * args.dt

        # 1. Forecast every member to the next time.
        for solver in members:
            solver.step_to(t_target)

        # Assimilate every `frequency` steps.
        if step % args.frequency != 0:
            continue

        # 2. Assemble forecast ensemble X (n_state x Ne).
        X = np.zeros((n_state, args.N))
        X[:N_PARAMS, :] = member_params.T
        X[N_PARAMS, :] = t_target
        for i, solver in enumerate(members):
            X[FIELD_OFFSET:, i] = solver.U[0]          # density = U[0]

        # 3. Observations from the exact solution at the current time.
        exact_rho = exact_density_on_cells(NOMINAL, t_target, n_field, x0=args.x0)
        y = exact_rho[cell_idx]

        prior_rho = X[FIELD_OFFSET:, :].mean(axis=1)
        prior_rmse = rmse(prior_rho, exact_rho)

        # 4. EnKF analysis -- root filter runs in C++ (Eigen) by default; the
        #    numpy implementation is kept as a cross-check / fallback.
        if args.filter == "cpp":
            X_a = enkf_filter_cpp(
                X, y, H, args.obs_error,
                state_loc=state_loc, obs_loc=obs_loc,
                num_globals=FIELD_OFFSET, loc_rad=args.loc_rad, localize=localize,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        else:
            X_a = enkf_analysis(
                X, y, H, args.obs_error, rng=rng,
                state_loc=state_loc, obs_loc=obs_loc,
                num_globals=FIELD_OFFSET, loc_rad=args.loc_rad, localize=localize,
            )
        post_rho = X_a[FIELD_OFFSET:, :].mean(axis=1)
        analysis_rmse = rmse(post_rho, exact_rho)

        # 5. Re-inject: analysis density, keep forecast velocity & pressure.
        for i, solver in enumerate(members):
            rho_a = np.maximum(X_a[FIELD_OFFSET:, i], 1e-6)  # keep positivity
            W = solver.primitive()                           # forecast rho,u,p
            solver.set_state_primitive(rho_a, W[1], W[2])
            member_params[i] = X_a[:N_PARAMS, i]             # track inferred ICs

        history.append((t_target, prior_rmse, analysis_rmse))
        param_history.append((t_target, *member_params.mean(axis=0)))
        last_prior_rho, last_post_rho, last_obs, last_t = prior_rho, post_rho, y, t_target

        inferred = member_params.mean(axis=0)
        print(f"[t={t_target:.2e}] prior_rmse={prior_rmse:.4e} "
              f"analysis_rmse={analysis_rmse:.4e}  "
              f"inferred p_high={inferred[0]:.3e} rho_high={inferred[2]:.4f}")

    return {
        "history": np.array(history),
        "param_history": np.array(param_history),
        "x_cells": x_cells,
        "exact_final": exact_density_on_cells(NOMINAL, last_t, n_field, x0=args.x0),
        "prior_final": last_prior_rho,
        "post_final": last_post_rho,
        "obs_x": cell_idx / n_field,   # whole cell points (i/N), not centers
        "obs_y": last_obs,
        "final_t": last_t,
    }


def write_outputs(res: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    np.savetxt(outdir / "cycle_density_final.txt",
               np.column_stack([res["x_cells"], res["exact_final"],
                                res["prior_final"], res["post_final"]]),
               header="x exact prior analysis", comments="# ")
    np.savetxt(outdir / "cycle_obs.txt",
               np.column_stack([res["obs_x"], res["obs_y"]]),
               header="x_obs y_obs", comments="# ")
    np.savetxt(outdir / "cycle_rmse.txt", res["history"],
               header="t prior_rmse analysis_rmse", comments="# ")
    print(f"[done] wrote density/obs/rmse tables to {outdir}/")


def make_plot(res: dict, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = res["history"]            # (cycles, 3): t, prior_rmse, analysis_rmse
    phist = res["param_history"]     # (cycles, 5): t, p_high, p_low, rho_high, rho_low
    x = res["x_cells"]
    exact, prior, post = res["exact_final"], res["prior_final"], res["post_final"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_d, ax_r), (ax_rmse, ax_p) = axes

    # (a) Density field at the final assimilation time.
    ax_d.plot(x, exact, color="black", lw=2.2, label="Exact (Sod)", zorder=3)
    ax_d.plot(x, prior, color="tab:blue", lw=1.6, ls="--",
              label="Forecast (prior) mean", zorder=2)
    ax_d.plot(x, post, color="tab:red", lw=1.8, label="Analysis mean", zorder=2)
    ax_d.scatter(res["obs_x"], res["obs_y"], color="gray", marker="o", s=14,
                 edgecolor="white", linewidth=0.4, zorder=5, label="Observations")
    ax_d.set_xlabel("x")
    ax_d.set_ylabel(r"density $\rho$")
    ax_d.set_title(f"Density at final t = {res['final_t']:.2e}")
    ax_d.legend(loc="upper right", framealpha=0.95)
    ax_d.grid(True, alpha=0.3)

    # (b) Residual (mean - exact) at the final time.
    ax_r.axhline(0.0, color="black", lw=0.8)
    ax_r.plot(x, prior - exact, color="tab:blue", lw=1.2, ls="--", label="prior - exact")
    ax_r.plot(x, post - exact, color="tab:red", lw=1.4, label="analysis - exact")
    ax_r.scatter(res["obs_x"], np.zeros_like(res["obs_x"]), color="gray",
                 marker="|", s=60, zorder=4)
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("residual")
    ax_r.set_title("Residual vs exact (final time)")
    ax_r.legend(loc="upper right", framealpha=0.95)
    ax_r.grid(True, alpha=0.3)

    # (c) RMSE vs time across the assimilation cycles.
    ax_rmse.plot(hist[:, 0], hist[:, 1], "o-", color="tab:blue", label="Prior (forecast)")
    ax_rmse.plot(hist[:, 0], hist[:, 2], "o-", color="tab:red", label="Analysis")
    ax_rmse.set_xlabel("t")
    ax_rmse.set_ylabel("density-field RMSE vs exact")
    ax_rmse.set_title("Assimilation cycles")
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)

    # (d) Inferred (augmented) parameters over time, normalised to truth.
    if phist.size:
        names = ["p_high", "p_low", "rho_high", "rho_low"]
        colors = ["tab:purple", "tab:green", "tab:orange", "tab:brown"]
        for k, (nm, col) in enumerate(zip(names, colors)):
            ax_p.plot(phist[:, 0], phist[:, 1 + k] / NOMINAL[k], "o-",
                      color=col, label=nm)
    ax_p.axhline(1.0, color="black", lw=0.8, ls=":")
    ax_p.set_xlabel("t")
    ax_p.set_ylabel("inferred / truth")
    ax_p.set_title("Augmented parameter inference (ensemble mean)")
    ax_p.legend(ncol=2, fontsize=9)
    ax_p.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot] wrote {save_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=20, help="ensemble size (Ne)")
    ap.add_argument("--frequency", type=int, default=10,
                    help="number of forecast micro-steps between DA updates")
    ap.add_argument("--dt", type=float, default=1.0e-5, help="forecast micro-step size")
    ap.add_argument("--t-final", dest="t_final", type=float, default=6.0e-4,
                    help="final time t_f")
    ap.add_argument("--nx", type=int, default=256, help="number of grid cells")
    ap.add_argument("--x0", type=float, default=0.5, help="diaphragm location")
    ap.add_argument("--obs-every", type=int, default=13,
                    help="even baseline: observe every k-th density cell")
    ap.add_argument("--obs-halfwin", dest="obs_halfwin", type=int, default=1,
                    help="cells observed on EACH side of every feature (jump straddle), "
                         "with feature locations fixed from the exact solution at t_final")
    ap.add_argument("--obs-error", type=float, default=0.01,
                    help="observation noise std (R = obs_error^2 I)")
    ap.add_argument("--p-sig", dest="p_sig", type=float, default=0.05,
                    help="initial log-normal spread for pressures")
    ap.add_argument("--rho-sig", dest="rho_sig", type=float, default=0.10,
                    help="initial log-normal spread for densities")
    ap.add_argument("--loc-rad", dest="loc_rad", type=float, default=20.0,
                    help="Gaspari-Cohn localization cutoff (cells)")
    ap.add_argument("--no-loc", action="store_true", help="disable localization")
    ap.add_argument("--filter", choices=["cpp", "python"], default="cpp",
                    help="root analysis backend: C++ EnKF (default) or numpy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "cycle_results"))
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = ap.parse_args()

    res = run_cycle_enkf(args)
    outdir = Path(args.outdir)
    write_outputs(res, outdir)
    if not args.no_plot:
        make_plot(res, outdir / "cycle_enkf.png")


if __name__ == "__main__":
    main()
