"""
Generate an ensemble of shock-tube runs by perturbing the left/right
pressures and densities, mirroring chord-mlef/euler-miniapp/MLEF/genEnsemble.sh.

Each (p_L, p_H, rho_L, rho_H) is perturbed log-normally:
    p_L^(m) = p_L * exp(P_SIG  * z),     z ~ N(0,1)
    rho_L^(m) = rho_L * exp(RHO_SIG * z),  etc.

Outputs (relative to --outdir, default 'ensemble/'):
    truth.npz                       — unperturbed reference run
    manifest.npz                    — sampled parameters for every member
    member_XXX/trajectory.npz       — per-member trajectory
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from sod_euler import EulerSolver1D


# ---------------------------------------------------------------------------
# Defaults that mirror genEnsemble.sh / ShockTube.input
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    N=30,
    seed=1,
    nx=256,
    xmin=0.0,
    xmax=1.0,
    gamma=1.4,
    p_high=1.0e5,
    p_low=1.0e4,
    rho_high=1.0,
    rho_low=0.125,
    x0=0.5,
    dt=1.0e-5,
    n_steps=60,
    plotting_interval=1,    # snapshots per dt, like Chord's plotting_interval=1
    p_sig=0.05,             # P_SIG  in genEnsemble.sh
    rho_sig=0.10,           # RHO_SIG
    cfl=0.5,
)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(params: dict, snapshot_every: int):
    """
    Run a single shock-tube simulation with the given parameters.
    Returns dict with arrays {x, t, U} where U has shape (n_snapshots, 3, nx).
    """
    solver = EulerSolver1D(
        nx=params["nx"],
        xmin=params["xmin"],
        xmax=params["xmax"],
        gamma=params["gamma"],
        cfl=params["cfl"],
    )
    solver.set_sod_like(
        rho_high=params["rho_high"], p_high=params["p_high"],
        rho_low=params["rho_low"],  p_low=params["p_low"],
        x0=params["x0"],
    )

    dt = params["dt"]
    n_steps = params["n_steps"]
    snaps_U, snaps_t = [solver.U.copy()], [solver.t]

    # March to fixed snapshot times t_k = k*dt using CFL-adaptive substeps.
    # This keeps snapshot times identical across all ensemble members and the
    # truth (required for MLEF) while preventing instability when perturbed
    # ICs push the sound speed past the fixed-dt CFL limit.
    for step in range(1, n_steps + 1):
        t_target = step * dt
        solver.step_to(t_target)
        if step % snapshot_every == 0 or step == n_steps:
            snaps_U.append(solver.U.copy())
            snaps_t.append(solver.t)

    return dict(
        x=solver.x.copy(),
        t=np.array(snaps_t),
        U=np.stack(snaps_U, axis=0),  # (n_snap, 3, nx)
    )


# ---------------------------------------------------------------------------
# Ensemble driver
# ---------------------------------------------------------------------------

def sample_perturbations(rng: np.random.Generator, base: dict,
                         p_sig: float, rho_sig: float):
    """Log-normal multiplicative perturbations on the four IC parameters."""
    z = rng.standard_normal(4)
    return dict(
        p_low   = base["p_low"]   * np.exp(p_sig   * z[0]),
        p_high  = base["p_high"]  * np.exp(p_sig   * z[1]),
        rho_low = base["rho_low"] * np.exp(rho_sig * z[2]),
        rho_high= base["rho_high"]* np.exp(rho_sig * z[3]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=DEFAULTS["N"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--nx", type=int, default=DEFAULTS["nx"])
    ap.add_argument("--p_sig", type=float, default=DEFAULTS["p_sig"])
    ap.add_argument("--rho_sig", type=float, default=DEFAULTS["rho_sig"])
    ap.add_argument("--outdir", default="ensemble")
    args = ap.parse_args()

    base = dict(DEFAULTS)
    base["nx"] = args.nx

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Truth run (unperturbed)
    # ------------------------------------------------------------------
    print(f"[truth] running unperturbed reference (nx={base['nx']}, "
          f"n_steps={base['n_steps']})")
    t0 = time.perf_counter()
    truth = run_one(base, snapshot_every=base["plotting_interval"])
    print(f"[truth] done in {time.perf_counter()-t0:.2f} s")

    np.savez(
        os.path.join(args.outdir, "truth.npz"),
        x=truth["x"], t=truth["t"], U=truth["U"],
        gamma=base["gamma"],
        p_low=base["p_low"], p_high=base["p_high"],
        rho_low=base["rho_low"], rho_high=base["rho_high"],
        x0=base["x0"], dt=base["dt"], n_steps=base["n_steps"],
    )

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    member_params = np.zeros((args.N, 4))   # columns: p_low, p_high, rho_low, rho_high

    for m in range(args.N):
        memdir = os.path.join(args.outdir, f"member_{m+1:03d}")
        os.makedirs(memdir, exist_ok=True)

        p = dict(base)
        p.update(sample_perturbations(rng, base, args.p_sig, args.rho_sig))
        member_params[m] = [p["p_low"], p["p_high"], p["rho_low"], p["rho_high"]]

        t0 = time.perf_counter()
        result = run_one(p, snapshot_every=base["plotting_interval"])
        elapsed = time.perf_counter() - t0

        np.savez(
            os.path.join(memdir, "trajectory.npz"),
            x=result["x"], t=result["t"], U=result["U"],
            gamma=p["gamma"],
            p_low=p["p_low"], p_high=p["p_high"],
            rho_low=p["rho_low"], rho_high=p["rho_high"],
            x0=p["x0"], dt=p["dt"], n_steps=p["n_steps"],
        )

        print(f"[m={m+1:03d}] pL={p['p_low']:.4e} pH={p['p_high']:.4e} "
              f"rL={p['rho_low']:.4f} rH={p['rho_high']:.4f}  "
              f"({elapsed:.2f} s)")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(args.outdir, "manifest.npz"),
        params=member_params,
        param_names=np.array(["p_low", "p_high", "rho_low", "rho_high"]),
        seed=args.seed,
        p_sig=args.p_sig,
        rho_sig=args.rho_sig,
        N=args.N,
        base_p_low=base["p_low"],
        base_p_high=base["p_high"],
        base_rho_low=base["rho_low"],
        base_rho_high=base["rho_high"],
    )

    print(f"\n[done] {args.N} members + truth saved under '{args.outdir}/'")


if __name__ == "__main__":
    main()
