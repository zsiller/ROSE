"""
Driver: solve the 1D shock tube using the same parameters as
chord-mlef/euler-miniapp/inputFiles/ShockTube.input, then plot the result.
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import h5py

from task_simulations.Shock_Tube.sod_euler import EulerSolver1D

def run_shock_tube(output_path: str, *params: float, x0: float = 0.5, n_steps: int = 60, snapshot_every: int = 1):

    solver = EulerSolver1D(nx=256, xmin=0.0, xmax=1.0, gamma=1.4, cfl=0.5)
    solver.set_sod_like(rho_high=params[2], p_high=params[0], rho_low=params[3], p_low=params[1], x0=x0)

    snaps_U, snaps_t = [solver.U.copy()], [solver.t]

    for step in range(1, n_steps + 1):
        t_target = step * 1.0e-5
        solver.step_to(t_target)
        if step % snapshot_every == 0 or step == n_steps:
            snaps_U.append(solver.U.copy())
            snaps_t.append(solver.t)

    W = solver.primitive()
    rho, u, p = W[0], W[1], W[2]
    a = np.sqrt(1.4 * p / rho)
    mach = u / a

    output_h5_path = os.path.join(output_path)
    with h5py.File(output_h5_path, "w") as f:
        # Write datasets
        f.create_dataset("x", data=solver.x)
        f.create_dataset("t", data=np.array(snaps_t))
        snaps_U_arr = np.array(snaps_U)  # shape: (n_snapshots, 3, nx)
        f.create_dataset("rho", data=snaps_U_arr[:, 0, :])      # density over time and space
        f.create_dataset("momentum", data=snaps_U_arr[:, 1, :]) # momentum over time and space
        f.create_dataset("energy", data=snaps_U_arr[:, 2, :])   # energy over time and space
 

        # Write metadata as attributes
        f.attrs["gamma"] = 1.4
        f.attrs["p_low"] = params[1]
        f.attrs["p_high"] = params[0]
        f.attrs["rho_low"] = params[3]
        f.attrs["rho_high"] = params[2]
        f.attrs["x0"] = x0
        f.attrs["n_steps"] = n_steps
    print(f"Wrote data to: {output_h5_path}")
    return output_h5_path



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--p-high", type=float, default=1.0e5)
    parser.add_argument("--p-low", type=float, default=1.0e4)
    parser.add_argument("--rho-high", type=float, default=1.0)
    parser.add_argument("--rho-low", type=float, default=0.125)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--n_steps", type=int, default=60)
    parser.add_argument("--snapshot-every", type=int, default=1)
    args = parser.parse_args()

    run_shock_tube(
        args.data_path,
        args.p_high, args.p_low, args.rho_high, args.rho_low,
        x0=args.x0,
        n_steps=args.n_steps,
        snapshot_every=args.snapshot_every,
    )
