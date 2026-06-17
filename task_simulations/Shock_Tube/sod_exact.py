#!/usr/bin/env python3
"""Exact (analytic) Sod shock-tube solution -- self-contained executable.

Solve the 1D Riemann problem for the Euler equations with a discontinuity at
x0 on the domain [0, 1]:

      left state  (x < x0):  rho = rho_high,  u = 0,  p = p_high
      right state (x > x0):  rho = rho_low,   u = 0,  p = p_low

and evaluate the *analytic* self-similar solution at any time t. Because the
solution is self-similar (a function of (x - x0)/t only), this is O(n_cells)
and effectively instantaneous for ANY t -- there is no time stepping. That is
the whole advantage over the numerical Euler solver: no iteration, no CFL, no
accumulation of discretization error. The fronts are perfectly sharp.

Wave structure (left -> right), for the standard high/low Sod setup:

    | left | <-rarefaction fan-> | region 3 | contact | region 2 | shock | right |
   xL5                          x53        x32                  x2R

    - xL5 : rarefaction head     (moves left at the left sound speed)
    - x53 : rarefaction tail
    - x32 : contact discontinuity (density jumps; u, p continuous)
    - x2R : shock front

Convention matches the rest of this repo (gamma = 1.4, x0 = 0.5, density at the
cell centers (i + 0.5)/n_cells).

Usage
-----
    ./sod_exact.py P_HIGH P_LOW RHO_HIGH RHO_LOW T [options]

    ./sod_exact.py 1e5 1e4 1.0 0.125 6e-4
    ./sod_exact.py 1e5 1e4 1.0 0.125 6e-4 --plot sod_exact.png
    ./sod_exact.py 1e5 1e4 1.0 0.125 6e-4 --out exact_state.npz --quiet
    # write an HDF5 in the sim task's format/naming (--h5 DIR auto-names the
    # file; --h5 FILE writes that exact path):
    ./sod_exact.py 1e5 1e4 1.0 0.125 6e-4 --h5 data/wf_0/
    ./sod_exact.py 1e5 1e4 1.0 0.125 6e-4 --h5 out.h5
    # initial state + 60 evenly-time-spaced snapshots (last = t) -> 61 rows:
    ./sod_exact.py 1e5 1e4 1.0 0.125 6e-4 --snapshot 60 --h5 traj.h5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

GAMMA = 1.4


# --------------------------------------------------------------------------- #
# Optional HDF5 output, matching the sim task's naming convention and format
# (task_simulations/file_namer.py + Shock_Tube/run_shock_tube.py): datasets
# x, t, rho, momentum, energy (the last three shaped (n_snap, nx)) plus the
# param/grid attributes. A single solution time is written as one snapshot
# (n_snap = 1), which construct_X/construct_Y read row-per-snapshot.
# --------------------------------------------------------------------------- #
def _h5_filename(params):
    """Canonical sim filename: shock_tube_exact__<params>__<timestamp>.h5.

    The run tag is intentionally omitted -- it lives in the directory path the
    caller passes, so it is not duplicated in the filename.
    """
    params_str = "_".join(f"{round(float(p), 5)}" for p in params)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"shock_tube_exact__{params_str}__{timestamp}.h5"


def write_h5(target, x, t, rho, momentum, energy, params, x0, n_steps=1):
    """Write a shock-tube HDF5 file in the sim task's format.

    `target` may be a directory (the canonical filename is generated inside it)
    or an explicit file path. Returns the path written.
    """
    import h5py  # lazy: keep the script importable without h5py

    if os.path.isdir(target) or target.endswith(os.sep):
        os.makedirs(target, exist_ok=True)
        path = os.path.join(target, _h5_filename(params))
    else:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        path = target

    rho = np.atleast_2d(rho)            # (n_snap, nx)
    momentum = np.atleast_2d(momentum)
    energy = np.atleast_2d(energy)
    t = np.atleast_1d(np.asarray(t, dtype=float))

    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.asarray(x, dtype=float))
        f.create_dataset("t", data=t)
        f.create_dataset("rho", data=rho)
        f.create_dataset("momentum", data=momentum)
        f.create_dataset("energy", data=energy)
        f.attrs["gamma"] = GAMMA
        f.attrs["p_high"] = float(params[0])
        f.attrs["p_low"] = float(params[1])
        f.attrs["rho_high"] = float(params[2])
        f.attrs["rho_low"] = float(params[3])
        f.attrs["x0"] = float(x0)
        f.attrs["n_steps"] = int(n_steps)
    return path


# --------------------------------------------------------------------------- #
# Time-independent Sod quantities (sound speeds, star pressure, region states).
# --------------------------------------------------------------------------- #
def _shock_residual(P, p_left, p_right, a_left, a_right, alpha):
    """Residual whose root P = p_star/p_right fixes the shock strength."""
    return (
        np.sqrt(2.0 / (GAMMA * (GAMMA - 1.0))) * (P - 1.0) / np.sqrt(1.0 + alpha * P)
        - (2.0 / (GAMMA - 1.0)) * (a_left / a_right)
        * (1.0 - ((p_right / p_left) * P) ** ((GAMMA - 1.0) / (2.0 * GAMMA)))
    )


def _find_P(p_left, p_right, a_left, a_right, alpha,
            lo=1.0 + 1e-9, hi=50.0, iters=200):
    """Bisection for the post-shock pressure ratio P = p_star / p_right."""
    f = lambda P: _shock_residual(P, p_left, p_right, a_left, a_right, alpha)
    a, b, fa = lo, hi, None
    fa = f(a)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0:
            return m
        if fa * fm < 0.0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def _shock_consts(p_high, p_low, rho_high, rho_low):
    """All time-independent quantities needed to assemble the solution."""
    a_left = np.sqrt(GAMMA * p_high / rho_high)    # left sound speed
    a_right = np.sqrt(GAMMA * p_low / rho_low)     # right sound speed
    alpha = (GAMMA + 1.0) / (GAMMA - 1.0)

    P = _find_P(p_high, p_low, a_left, a_right, alpha)
    p_star = p_low * P                              # pressure in regions 2 & 3
    # post-shock density (region 2), Rankine-Hugoniot across the shock
    dens_2 = ((1.0 + alpha * P) / (alpha + P)) * rho_low
    # contact velocity (= u in regions 2 and 3)
    V = (2.0 / (GAMMA - 1.0)) * a_left \
        * (1.0 - (p_star / p_high) ** ((GAMMA - 1.0) / (2.0 * GAMMA)))
    # post-rarefaction density (region 3), isentropic from the left state
    dens_3 = rho_high * (p_star / p_high) ** (1.0 / GAMMA)
    # shock speed
    C = ((P - 1.0) * a_right ** 2) / (GAMMA * V)
    return dict(a_left=a_left, V=V, C=C, p_star=p_star,
                dens_2=dens_2, dens_3=dens_3)


def exact_state(p_high, p_low, rho_high, rho_low, t, n_cells=256, x0=0.5):
    """Full exact primitive solution (rho, u, p) at the cell centers and time t.

    Returns
    -------
    x   : (n_cells,) cell-center coordinates
    rho : (n_cells,) density
    u   : (n_cells,) velocity
    p   : (n_cells,) pressure
    feat: dict with the four feature x-locations and region constants
    """
    x = (np.arange(n_cells, dtype=float) + 0.5) / n_cells
    rho = np.empty(n_cells)
    u = np.empty(n_cells)
    p = np.empty(n_cells)

    if t <= 0.0:  # initial step function
        left = x < x0
        rho = np.where(left, rho_high, rho_low)
        u = np.zeros(n_cells)
        p = np.where(left, p_high, p_low)
        feat = dict(xL5=x0, x53=x0, x32=x0, x2R=x0)
        return x, rho, u, p, feat

    s = _shock_consts(p_high, p_low, rho_high, rho_low)
    a_left, V, C = s["a_left"], s["V"], s["C"]
    p_star, dens_2, dens_3 = s["p_star"], s["dens_2"], s["dens_3"]

    # Feature locations at time t (ascending in x).
    xL5 = x0 - a_left * t                                   # rarefaction head
    x53 = x0 + ((V * (GAMMA + 1.0) / 2.0) - a_left) * t     # rarefaction tail
    x32 = x0 + V * t                                        # contact
    x2R = x0 + C * t                                        # shock

    # Self-similar rarefaction fan (region 5).
    xi = (x - x0) / t
    u5 = (2.0 / (GAMMA + 1.0)) * (xi + a_left)
    a5 = u5 - xi
    a5_safe = np.where(a5 > 0.0, a5, 1.0)
    p5 = p_high * (a5_safe / a_left) ** ((2.0 * GAMMA) / (GAMMA - 1.0))
    dens_5 = GAMMA * p5 / a5_safe ** 2

    # Assemble region by region (right -> left, overwriting).
    rho[:] = rho_low;       u[:] = 0.0;  p[:] = p_low          # right state
    m2 = x <= x2R
    rho[m2] = dens_2;       u[m2] = V;   p[m2] = p_star        # region 2
    m3 = x <= x32
    rho[m3] = dens_3;       u[m3] = V;   p[m3] = p_star        # region 3
    m5 = (x > xL5) & (x <= x53)
    rho[m5] = dens_5[m5];   u[m5] = u5[m5];  p[m5] = p5[m5]    # rarefaction
    mL = x <= xL5
    rho[mL] = rho_high;     u[mL] = 0.0;  p[mL] = p_high       # left state

    feat = dict(xL5=float(xL5), x53=float(x53), x32=float(x32), x2R=float(x2R),
                V=float(V), C=float(C), p_star=float(p_star),
                dens_2=float(dens_2), dens_3=float(dens_3))
    return x, rho, u, p, feat


# --------------------------------------------------------------------------- #
# Array-of-params conveniences, the in-process API the assimilation / inference
# code calls (params = [p_high, p_low, rho_high, rho_low]).
# --------------------------------------------------------------------------- #
def shock_features(params, t, x0=0.5):
    """x-locations of the four features at time t, given the 4 IC params:

    ``(xL5, x53, x32, x2R)`` = rarefaction head, rarefaction tail, contact, shock.
    """
    p = np.asarray(params, dtype=float)
    if t <= 0.0:
        return x0, x0, x0, x0
    s = _shock_consts(p[0], p[1], p[2], p[3])
    xL5 = x0 - s["a_left"] * t
    x53 = x0 + ((s["V"] * (GAMMA + 1.0) / 2.0) - s["a_left"]) * t
    x32 = x0 + s["V"] * t
    x2R = x0 + s["C"] * t
    return float(xL5), float(x53), float(x32), float(x2R)


def exact_density_on_cells(params, t, n_cells=256, x0=0.5):
    """Exact Sod density at the cell centers ``(i + 0.5)/n_cells`` at time t,
    for IC ``params = [p_high, p_low, rho_high, rho_low]``."""
    p = np.asarray(params, dtype=float)
    _, rho, _, _, _ = exact_state(p[0], p[1], p[2], p[3], t, n_cells, x0)
    return rho


# --------------------------------------------------------------------------- #
# Snapshots: conservative-field trajectory evenly spaced in time, for the sim
# HDF5 format. Mirrors sod_euler's --snapshot, but trivially -- the exact
# solution is analytic, so each snapshot is a direct evaluation at its time (no
# stepping). t_start is always 0 here (no restart), so the initial state is the
# IC step function.
# --------------------------------------------------------------------------- #
def exact_snapshots(p_high, p_low, rho_high, rho_low, t_target, n_snapshots,
                    n_cells=256, x0=0.5):
    """Return (times, rho, momentum, energy) for K+1 snapshots evenly spaced in
    time: the INITIAL state (t=0) plus K states at ``k*t_target/K`` for
    ``k = 1 .. K``, the last exactly at ``t_target``. Each field is (K+1, nx).
    """
    K = int(n_snapshots)
    times = [0.0] + [t_target * k / K for k in range(1, K + 1)]
    rho_rows, mom_rows, ener_rows = [], [], []
    for tk in times:
        _, rho_k, u_k, p_k, _ = exact_state(p_high, p_low, rho_high, rho_low,
                                            tk, n_cells, x0)
        rho_rows.append(rho_k)
        mom_rows.append(rho_k * u_k)
        ener_rows.append(p_k / (GAMMA - 1.0) + 0.5 * rho_k * u_k * u_k)
    return (np.array(times), np.array(rho_rows),
            np.array(mom_rows), np.array(ener_rows))


# --------------------------------------------------------------------------- #
# Plot.
# --------------------------------------------------------------------------- #
def make_plot(x, rho, u, p, feat, t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             constrained_layout=True)
    fields = [("density  $\\rho$", rho, "tab:blue"),
              ("velocity  $u$", u, "tab:green"),
              ("pressure  $p$", p, "tab:red")]
    feats = [("xL5", "rarefaction head", "0.5"),
             ("x53", "rarefaction tail", "0.5"),
             ("x32", "contact", "tab:purple"),
             ("x2R", "shock", "tab:orange")]
    for ax, (label, fld, c) in zip(axes, fields):
        ax.plot(x, fld, color=c, lw=2.0)
        for key, name, fc in feats:
            ax.axvline(feat[key], color=fc, ls="--", lw=1.0, alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    # annotate features on the top panel only
    ymax = rho.max()
    for key, name, fc in feats:
        axes[0].annotate(name, xy=(feat[key], ymax), rotation=90,
                         va="top", ha="right", fontsize=8, color=fc)
    axes[-1].set_xlabel("x")
    axes[0].set_title(f"Exact Sod solution at t = {t:.4e}  "
                      f"(analytic, sharp fronts)")
    fig.savefig(path, dpi=150)
    print(f"[plot] wrote {path}")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("p_high", type=float, help="left  pressure")
    ap.add_argument("p_low", type=float, help="right pressure")
    ap.add_argument("rho_high", type=float, help="left  density")
    ap.add_argument("rho_low", type=float, help="right density")
    ap.add_argument("t", type=float, help="solution time")
    ap.add_argument("--nx", type=int, default=256, help="number of cells on [0,1]")
    ap.add_argument("--x0", type=float, default=0.5, help="initial discontinuity")
    ap.add_argument("--out", default=None, help="save (x, rho, u, p) to this .npz")
    ap.add_argument("--h5", default=None,
                    help="write an HDF5 file in the sim task's format/naming; value "
                         "is a directory (canonical filename auto-generated) or a "
                         "full file path")
    ap.add_argument("--snapshot", type=int, default=None, metavar="K",
                    help="write the INITIAL state (t=0) plus K states evenly spaced "
                         "in TIME over (0, t] into the HDF5 -> K+1 snapshots, "
                         "rho/momentum/energy become (K+1, nx) and t holds their "
                         "times, last exactly at t. Without the flag, only the state "
                         "at t is written. Matches sod_euler.py's --snapshot.")
    ap.add_argument("--plot", default=None, help="write a 3-panel figure to this path")
    ap.add_argument("--quiet", action="store_true", help="suppress the field summary")
    args = ap.parse_args(argv)

    x, rho, u, p, feat = exact_state(args.p_high, args.p_low,
                                     args.rho_high, args.rho_low,
                                     args.t, n_cells=args.nx, x0=args.x0)

    if not args.quiet:
        print(f"[exact] params=[p_hi={args.p_high:.4e}, p_lo={args.p_low:.4e}, "
              f"rho_hi={args.rho_high:.4e}, rho_lo={args.rho_low:.4e}]  "
              f"t={args.t:.4e}  nx={args.nx}")
        if args.t > 0:
            print(f"[features] rarefaction head x={feat['xL5']:.4f}  "
                  f"tail x={feat['x53']:.4f}  contact x={feat['x32']:.4f}  "
                  f"shock x={feat['x2R']:.4f}")
        print(f"[fields] rho in [{rho.min():.4f}, {rho.max():.4f}]  "
              f"u in [{u.min():.3f}, {u.max():.3f}]  "
              f"p in [{p.min():.3e}, {p.max():.3e}]")

    params = [args.p_high, args.p_low, args.rho_high, args.rho_low]
    if args.out:
        np.savez(args.out, x=x, rho=rho, u=u, p=p, t=args.t,
                 params=np.array(params))
        print(f"[out] wrote {args.out}")
    if args.h5:
        # conservative fields for the sim format: momentum = rho*u, energy = E
        if args.snapshot:
            t_arr, traj_rho, traj_mom, traj_ener = exact_snapshots(
                args.p_high, args.p_low, args.rho_high, args.rho_low,
                args.t, args.snapshot, n_cells=args.nx, x0=args.x0)
            path = write_h5(args.h5, x, t_arr, traj_rho, traj_mom, traj_ener,
                            params, args.x0, n_steps=t_arr.size)
            print(f"[h5] wrote {path}  ({t_arr.size} snapshot(s))")
        else:
            momentum = rho * u
            energy = p / (GAMMA - 1.0) + 0.5 * rho * u * u
            path = write_h5(args.h5, x, args.t, rho, momentum, energy,
                            params, args.x0)
            print(f"[h5] wrote {path}  (1 snapshot)")
    if args.plot:
        make_plot(x, rho, u, p, feat, args.t, args.plot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
