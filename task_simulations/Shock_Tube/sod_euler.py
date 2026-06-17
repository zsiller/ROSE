#!/usr/bin/env python3
"""Numerical Euler shock-tube solver -- self-contained executable.

Solve the 1D Euler equations  U_t + F(U)_x = 0,  U = (rho, rho*u, E)  on [0, 1]
with a finite-volume scheme:

    - MUSCL reconstruction (minmod limiter) in primitive variables
    - HLLC approximate Riemann solver at the cell faces
    - SSPRK2 (Heun) time integration, 2nd order in space and time
    - transmissive (zero-gradient) boundaries

Unlike the analytic solver (``sod_exact.py``), this MARCHES IN TIME with a
CFL-limited step, so reaching a late time t requires many substeps -- the cost
grows ~linearly with t. The discrete shock/contact are therefore smeared over a
few cells (numerical diffusion), which is exactly the discretization error the
exact solver avoids.

Performance
-----------
* Everything is vectorized over cells (numpy); a step is a handful of array ops.
* ``step_to(t)`` takes CFL substeps and shortens the final one to land exactly
  on t -- so snapshots across an ensemble are at a consistent time.
* Number of substeps ~ t * (|u| + a)_max / (cfl * dx). To reach a *late* time,
  either pass a large t (it will iterate internally) or checkpoint:

Snapshots
---------
* ``--snapshot K`` writes the INITIAL state plus K states evenly spaced in TIME
  over (t_start, t] into the HDF5 trajectory -> K+1 snapshots, exactly like the
  sim task's files: rho/momentum/energy become (K+1, nx) and the ``t`` dataset
  lists their times, the first at ``t_start`` and the last exactly at ``t``. It
  is a COUNT, not a substep stride -- the CFL substeps run hidden inside each
  interval, so the snapshot count is independent of ``cfl`` / ``nx``. This
  matches ``run_shock_tube.py`` exactly: its 61 is the initial state + 60
  fixed-time targets, i.e. ``--snapshot 60``. Without the flag, only the final
  state is written (one snapshot).

Restart capability
------------------
* ``--save STATE.h5`` writes the state as a sim-format HDF5 file (the same format
  as ``--h5``); its conservative datasets rho/momentum/energy ARE the field U, so
  the file is a complete checkpoint -- no separate format needed. With
  ``--snapshot`` the file holds the whole trajectory; restart resumes from the
  LAST snapshot in it.
* ``--restart STATE.h5`` loads such a file and CONTINUES from its (latest)
  snapshot time up to the requested t (the 4 positional params are then ignored
  for the initial condition -- the state comes from the file). Any sim-format
  HDF5 works, including a full campaign simulation output, so you can reach very
  late times in stages or branch many runs from one expensive prefix.

  Note: because CFL substeps are re-partitioned from the checkpoint, a restarted
  run matches a single-shot run to discretization accuracy (a per-step
  truncation), not bit-for-bit. The CFL number is not part of the saved state;
  it is taken from ``--cfl`` (default 0.5) on restart.

Usage
-----
    ./sod_euler.py P_HIGH P_LOW RHO_HIGH RHO_LOW T [options]

    ./sod_euler.py 1e5 1e4 1.0 0.125 6e-4
    ./sod_euler.py 1e5 1e4 1.0 0.125 6e-4 --plot sod_euler.png
    # write an HDF5 in the sim task's format/naming (--h5 DIR auto-names the
    # file; --h5 FILE writes that exact path):
    ./sod_euler.py 1e5 1e4 1.0 0.125 6e-4 --h5 data/wf_0/
    ./sod_euler.py 1e5 1e4 1.0 0.125 6e-4 --h5 out.h5
    # initial state + 60 evenly-time-spaced snapshots (last = t) -> 61 rows,
    # matching run_shock_tube.py:
    ./sod_euler.py 1e5 1e4 1.0 0.125 6e-4 --snapshot 60 --h5 traj.h5
    # checkpoint at 3e-4, then resume to 6e-4 (HDF5 checkpoint):
    ./sod_euler.py 1e5 1e4 1.0 0.125 3e-4 --save ckpt.h5
    ./sod_euler.py 1e5 1e4 1.0 0.125 6e-4 --restart ckpt.h5 --plot resumed.png
    # a file written by --h5 (or a campaign sim output) restarts too:
    ./sod_euler.py 1e5 1e4 1.0 0.125 6e-4 --restart data/shock_tube__...h5
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
# param/grid attributes. The final state is written as one snapshot
# (n_snap = 1); the conservative field U maps straight onto the datasets.
# --------------------------------------------------------------------------- #
def _h5_filename(params):
    """Canonical sim filename: shock_tube_euler__<params>__<timestamp>.h5.

    The run tag is intentionally omitted -- it lives in the directory path the
    caller passes, so it is not duplicated in the filename.
    """
    params_str = "_".join(f"{round(float(p), 5)}" for p in params)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"shock_tube_euler__{params_str}__{timestamp}.h5"


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
# Primitive <-> conservative and the physical flux.
# --------------------------------------------------------------------------- #
def cons_to_prim(U, gamma):
    """U = [rho, rho*u, E] -> W = [rho, u, p]."""
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
    return np.stack([rho, u, p], axis=0)


def prim_to_cons(W, gamma):
    """W = [rho, u, p] -> U = [rho, rho*u, E]."""
    rho, u, p = W[0], W[1], W[2]
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, E], axis=0)


def euler_flux(U, gamma):
    """Physical flux F(U)."""
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
    F = np.empty_like(U)
    F[0] = rho * u
    F[1] = rho * u * u + p
    F[2] = u * (U[2] + p)
    return F


# --------------------------------------------------------------------------- #
# MUSCL reconstruction and HLLC flux.
# --------------------------------------------------------------------------- #
def minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def muscl_reconstruct(W):
    """Left/right primitive states at interior faces. W has 2 ghosts per side."""
    dW_left = W[:, 1:-1] - W[:, :-2]
    dW_right = W[:, 2:] - W[:, 1:-1]
    slope = minmod(dW_left, dW_right)
    WL = W[:, 1:-2] + 0.5 * slope[:, :-1]
    WR = W[:, 2:-1] - 0.5 * slope[:, 1:]
    return WL, WR


def hllc_flux(WL, WR, gamma):
    """HLLC flux at faces (Toro, Riemann Solvers, 3rd ed., sec. 10.6)."""
    rhoL, uL, pL = WL[0], WL[1], WL[2]
    rhoR, uR, pR = WR[0], WR[1], WR[2]

    rhoL = np.maximum(rhoL, 1e-12); rhoR = np.maximum(rhoR, 1e-12)
    pL = np.maximum(pL, 1e-12);     pR = np.maximum(pR, 1e-12)

    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    rho_bar = 0.5 * (rhoL + rhoR)
    a_bar = 0.5 * (aL + aR)
    p_pvrs = 0.5 * (pL + pR) - 0.5 * (uR - uL) * rho_bar * a_bar
    p_star = np.maximum(0.0, p_pvrs)

    qL = np.where(p_star <= pL, 1.0,
                  np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / pL - 1.0)))
    qR = np.where(p_star <= pR, 1.0,
                  np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / pR - 1.0)))
    SL = uL - aL * qL
    SR = uR + aR * qR

    num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den = rhoL * (SL - uL) - rhoR * (SR - uR)
    S_star = num / den

    UL = prim_to_cons(WL, gamma)
    UR = prim_to_cons(WR, gamma)
    FL = euler_flux(UL, gamma)
    FR = euler_flux(UR, gamma)

    def star_state(U, W, S, Ss):
        rho, u, p = W[0], W[1], W[2]
        factor = rho * (S - u) / (S - Ss)
        E = U[2]
        Us = np.empty_like(U)
        Us[0] = factor
        Us[1] = factor * Ss
        Us[2] = factor * (E / rho + (Ss - u) * (Ss + p / (rho * (S - u))))
        return Us

    UL_star = star_state(UL, WL, SL, S_star)
    UR_star = star_state(UR, WR, SR, S_star)
    F_star_L = FL + SL * (UL_star - UL)
    F_star_R = FR + SR * (UR_star - UR)

    return np.where(SL >= 0.0, FL,
           np.where(S_star >= 0.0, F_star_L,
           np.where(SR >= 0.0, F_star_R, FR)))


# --------------------------------------------------------------------------- #
# Solver.
# --------------------------------------------------------------------------- #
class EulerSolver1D:
    def __init__(self, nx, xmin=0.0, xmax=1.0, gamma=GAMMA, cfl=0.5):
        self.nx = nx
        self.xmin = xmin
        self.xmax = xmax
        self.gamma = gamma
        self.cfl = cfl
        self.dx = (xmax - xmin) / nx
        self.x = xmin + (np.arange(nx) + 0.5) * self.dx
        self.U = np.zeros((3, nx))
        self.t = 0.0

    def set_sod_like(self, rho_high, p_high, rho_low, p_low, x0=0.5):
        left = self.x < x0
        rho = np.where(left, rho_high, rho_low)
        u = np.zeros(self.nx)
        p = np.where(left, p_high, p_low)
        self.U = prim_to_cons(np.stack([rho, u, p], axis=0), self.gamma)
        self.t = 0.0

    def primitive(self):
        return cons_to_prim(self.U, self.gamma)

    def set_state_primitive(self, rho, u, p):
        """Inject a (possibly assimilated) primitive state without touching t."""
        self.U = prim_to_cons(np.stack([rho, u, p], axis=0), self.gamma)

    def _add_ghost_cells(self, U):
        Ug = np.empty((3, U.shape[1] + 4))
        Ug[:, 2:-2] = U
        Ug[:, 0] = U[:, 0]; Ug[:, 1] = U[:, 0]
        Ug[:, -1] = U[:, -1]; Ug[:, -2] = U[:, -1]
        return Ug

    def _max_wave_speed(self, U):
        W = cons_to_prim(U, self.gamma)
        rho, u, p = W[0], W[1], W[2]
        a = np.sqrt(self.gamma * np.maximum(p, 1e-12) / np.maximum(rho, 1e-12))
        return float(np.max(np.abs(u) + a))

    def rhs(self, U):
        Ug = self._add_ghost_cells(U)
        Wg = cons_to_prim(Ug, self.gamma)
        WL, WR = muscl_reconstruct(Wg)
        F = hllc_flux(WL, WR, self.gamma)
        return -(F[:, 1:] - F[:, :-1]) / self.dx

    def step(self, dt):
        U0 = self.U
        k1 = self.rhs(U0)
        U1 = U0 + dt * k1
        k2 = self.rhs(U1)
        self.U = 0.5 * (U0 + U1 + dt * k2)
        self.t += dt

    def step_to(self, t_target):
        """Advance to t_target with CFL substeps; returns the substep count."""
        n_sub = 0
        while self.t < t_target - 1e-14:
            smax = self._max_wave_speed(self.U)
            sub_dt = min(self.cfl * self.dx / max(smax, 1e-12), t_target - self.t)
            self.step(sub_dt)
            n_sub += 1
        return n_sub

    def run_with_snapshots(self, t_target, n_snapshots=None):
        """Advance to t_target, recording states evenly spaced in TIME.

        * ``n_snapshots is None`` -> a single snapshot at the final time
          (the default: just the end state).
        * ``n_snapshots = K`` -> the INITIAL state plus K states evenly spaced in
          time at ``t_start + k*(t_target - t_start)/K`` for ``k = 1 .. K`` --
          so K+1 snapshots, the first at ``t_start`` and the last exactly at
          ``t_target``. This mirrors ``run_shock_tube.py`` (its 61 = initial + 60
          fixed-time targets). It is a COUNT, not a substep stride: the CFL
          substeps run hidden inside each interval, so the snapshot count is
          independent of ``cfl`` / ``nx``.

        Returns ``(times, states, n_sub)``: ``states[k]`` is the (3, nx)
        conservative field U at ``times[k]``; ``n_sub`` is the total CFL substeps.
        """
        if not n_snapshots:
            n_sub = self.step_to(t_target)
            return [self.t], [self.U.copy()], n_sub

        K = int(n_snapshots)
        t0 = self.t
        times = [self.t]                  # include the initial state
        states = [self.U.copy()]
        n_sub = 0
        for k in range(1, K + 1):
            t_k = t0 + (t_target - t0) * k / K
            n_sub += self.step_to(t_k)
            times.append(self.t)
            states.append(self.U.copy())
        return times, states, n_sub

    # -- restart from an HDF5 checkpoint ------------------------------------
    @classmethod
    def load(cls, path, cfl=0.5):
        """Rebuild a solver from a sim-format HDF5 file (see write_h5).

        The conservative datasets rho/momentum/energy ARE the state U, so any
        such file is a complete checkpoint. If it holds several snapshots, the
        LAST (latest-time) one is used -- so a full campaign sim output restarts
        just as well as a single-state file written here. The grid is recovered
        from `x`; `cfl` is a solver control (not stored in the format) and is
        taken from the caller (default 0.5).
        """
        import h5py  # lazy: only needed when restarting

        with h5py.File(path, "r") as f:
            x = np.asarray(f["x"], dtype=float)
            t = np.asarray(f["t"], dtype=float)
            rho = np.atleast_2d(np.asarray(f["rho"], dtype=float))
            mom = np.atleast_2d(np.asarray(f["momentum"], dtype=float))
            ener = np.atleast_2d(np.asarray(f["energy"], dtype=float))
            gamma = float(f.attrs.get("gamma", GAMMA))

        nx = x.size
        dx = float(x[1] - x[0])               # x = xmin + (i + 0.5)*dx
        xmin = float(x[0] - 0.5 * dx)
        xmax = xmin + nx * dx
        solver = cls(nx, xmin, xmax, gamma, cfl)
        solver.U = np.stack([rho[-1], mom[-1], ener[-1]], axis=0)
        solver.t = float(t[-1])
        return solver


# --------------------------------------------------------------------------- #
# Plot.
# --------------------------------------------------------------------------- #
def make_plot(solver, t, path, n_sub, t_start):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W = solver.primitive()
    rho, u, p = W[0], W[1], W[2]
    cells = np.arange(solver.nx)        # plot over grid cell index, not physical x
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             constrained_layout=True)
    for ax, (label, fld, c) in zip(
            axes, [("density  $\\rho$", rho, "tab:blue"),
                   ("velocity  $u$", u, "tab:green"),
                   ("pressure  $p$", p, "tab:red")]):
        ax.plot(cells, fld, color=c, lw=1.6, ms=3)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("grid cell index")
    start_note = f"from t={t_start:.2e} (restart)  " if t_start > 0 else ""
    axes[0].set_title(f"Euler (MUSCL-HLLC) solution at t = {t:.4e}  "
                      f"[{start_note}{n_sub} CFL substeps, nx={solver.nx}]")
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
    ap.add_argument("t", type=float, help="target solution time")
    ap.add_argument("--nx", type=int, default=256, help="number of cells on [0,1]")
    ap.add_argument("--x0", type=float, default=0.5, help="initial discontinuity")
    ap.add_argument("--cfl", type=float, default=0.5, help="CFL number")
    ap.add_argument("--restart", default=None,
                    help="resume from this sim-format HDF5 file (IC params then "
                         "ignored; latest snapshot is used as the state)")
    ap.add_argument("--save", default=None,
                    help="write the final state as a sim-format HDF5 for later "
                         "restart; value is a directory (canonical name) or a file "
                         "path. Any file from --save or --h5 can be passed to --restart")
    ap.add_argument("--out", default=None, help="save (x, rho, u, p) to this .npz")
    ap.add_argument("--h5", default=None,
                    help="write an HDF5 file in the sim task's format/naming; value "
                         "is a directory (canonical filename auto-generated) or a "
                         "full file path")
    ap.add_argument("--snapshot", type=int, default=None, metavar="K",
                    help="write the INITIAL state plus K states evenly spaced in "
                         "TIME over (t_start, t] into the HDF5 trajectory -> K+1 "
                         "snapshots, rho/momentum/energy become (K+1, nx) and t holds "
                         "their times, first at t_start and last exactly at t. A "
                         "count, not a substep stride: independent of cfl/nx, "
                         "matching run_shock_tube.py (its 61 = initial + 60). Without "
                         "the flag, only the final state is written. Affects --h5 and "
                         "--save; restart resumes from the last snapshot in the file.")
    ap.add_argument("--plot", default=None, help="write a 3-panel figure to this path")
    ap.add_argument("--quiet", action="store_true", help="suppress the field summary")
    args = ap.parse_args(argv)

    # --- build or resume the solver ----------------------------------------
    if args.restart:
        solver = EulerSolver1D.load(args.restart, cfl=args.cfl)
        t_start = solver.t
        if args.t < t_start - 1e-14:
            ap.error(f"target t={args.t:.4e} is before the checkpoint "
                     f"time t={t_start:.4e}; nothing to do.")
        if not args.quiet:
            print(f"[restart] loaded {args.restart}  t_start={t_start:.4e}  "
                  f"nx={solver.nx}  -> advancing to t={args.t:.4e}")
    else:
        solver = EulerSolver1D(args.nx, 0.0, 1.0, GAMMA, args.cfl)
        solver.set_sod_like(rho_high=args.rho_high, p_high=args.p_high,
                            rho_low=args.rho_low, p_low=args.p_low, x0=args.x0)
        t_start = 0.0

    # --- march in time -----------------------------------------------------
    wall0 = time.perf_counter()
    snaps_t, snaps_U, n_sub = solver.run_with_snapshots(args.t, args.snapshot)
    wall = time.perf_counter() - wall0

    # Trajectory stacked as (n_snap, nx) per field for the sim HDF5 format.
    snaps_arr = np.array(snaps_U)                  # (n_snap, 3, nx)
    t_arr = np.array(snaps_t)                      # (n_snap,)
    traj_rho, traj_mom, traj_ener = snaps_arr[:, 0], snaps_arr[:, 1], snaps_arr[:, 2]

    W = solver.primitive()
    rho, u, p = W[0], W[1], W[2]
    if not args.quiet:
        snap_note = f"  snapshots={t_arr.size}" if args.snapshot else ""
        print(f"[euler] t={solver.t:.4e}  substeps={n_sub}{snap_note}  "
              f"wall={wall * 1e3:.1f} ms  nx={solver.nx}  cfl={args.cfl}")
        print(f"[fields] rho in [{rho.min():.4f}, {rho.max():.4f}]  "
              f"u in [{u.min():.3f}, {u.max():.3f}]  "
              f"p in [{p.min():.3e}, {p.max():.3e}]")

    # --- outputs -----------------------------------------------------------
    params = [args.p_high, args.p_low, args.rho_high, args.rho_low]
    if args.save:
        path = write_h5(args.save, solver.x, t_arr,
                        traj_rho, traj_mom, traj_ener,
                        params, args.x0, n_steps=n_sub)
        print(f"[save] HDF5 checkpoint written to {path}  "
              f"({t_arr.size} snapshot(s), t={solver.t:.4e})")
    if args.out:
        np.savez(args.out, x=solver.x, rho=rho, u=u, p=p, t=solver.t,
                 params=np.array(params))
        print(f"[out] wrote {args.out}")
    if args.h5:
        # conservative field U = [rho, momentum, energy] maps onto the datasets
        path = write_h5(args.h5, solver.x, t_arr,
                        traj_rho, traj_mom, traj_ener,
                        params, args.x0, n_steps=n_sub)
        print(f"[h5] wrote {path}  ({t_arr.size} snapshot(s))")
    if args.plot:
        make_plot(solver, solver.t, args.plot, n_sub, t_start)

    return 0


if __name__ == "__main__":
    sys.exit(main())
