"""Space-time density maps (analytical vs numerical) for one shock-tube sample.

For the sample below (a set of Sod ICs) this builds the density field rho(x, t)
from both the exact Sod solution and the MUSCL-HLLC numerical solver, then plots
the two as 2D maps over (x, t) side by side in one figure, sharing a single
colorbar so the analytical and numerical fields are directly comparable.

Edit SAMPLE below: [p_high, p_low, rho_high, rho_low].
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "task_simulations" / "Shock_Tube"))

from sod_euler import EulerSolver1D          # noqa: E402
from sod_exact import exact_density_on_cells  # noqa: E402

# --- inputs -----------------------------------------------------------------
# [p_high, p_low, rho_high, rho_low]
SAMPLE = np.array([1.0e5, 1.0e4, 1.0, 0.125])

T_FINAL = 6.0e-4   # final snapshot time
N_CELLS = 256      # uniform cells over [0, 1]
N_TIMES = 200      # time samples (excludes t=0, where exact is a step)
GAMMA = 1.4
CFL = 0.5


def density_maps(params):
    """Return (x, t, rho_exact, rho_num) space-time fields, shape (N_TIMES, N_CELLS)."""
    p = np.asarray(params, float)
    p_high, p_low, rho_high, rho_low = p

    x = (np.arange(N_CELLS) + 0.5) / N_CELLS
    times = np.linspace(T_FINAL / N_TIMES, T_FINAL, N_TIMES)

    solver = EulerSolver1D(nx=N_CELLS, xmin=0.0, xmax=1.0, gamma=GAMMA, cfl=CFL)
    solver.set_sod_like(rho_high=rho_high, p_high=p_high, rho_low=rho_low, p_low=p_low, x0=0.5)

    rho_exact = np.empty((N_TIMES, N_CELLS))
    rho_num = np.empty((N_TIMES, N_CELLS))
    for j, t in enumerate(times):
        solver.step_to(t)
        rho_num[j] = solver.primitive()[0]
        rho_exact[j] = exact_density_on_cells(params, t, N_CELLS)
    return x, times, rho_exact, rho_num


# --- build the sample -------------------------------------------------------
x, t, rho_exact, rho_num = density_maps(SAMPLE)

vmin = min(rho_exact.min(), rho_num.min())
vmax = max(rho_exact.max(), rho_num.max())

# --- plot -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
extent = [0.0, 1.0, t[0] * 1e3, T_FINAL * 1e3]  # time axis in ms

for ax, field, label in zip(axes, [rho_exact, rho_num],
                            ["Analytical (exact Sod)", "Numerical (MUSCL-HLLC)"]):
    im = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                   cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(label)
    ax.set_xlabel("x")
axes[0].set_ylabel("t  [ms]")

cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label(r"$\rho$")

fig.suptitle(rf"Shock-tube density: $p_H{{=}}{SAMPLE[0]:g},\ p_L{{=}}{SAMPLE[1]:g},\ "
             rf"\rho_H{{=}}{SAMPLE[2]:g},\ \rho_L{{=}}{SAMPLE[3]:g}$")

out = _ROOT / "figures" / "shock_tube_density_maps.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
