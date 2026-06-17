"""Plot exact (analytic) vs numerical shock-tube density at the final time.

Default Sod operating point from EnKF/EnKF_driver.py:
    TRUTH   = [p_high, p_low, rho_high, rho_low] = [1e5, 1e4, 1.0, 0.125]
    T_FINAL = 6.0e-4, on 256 uniform cells over [0, 1].

Produces one figure with two separate panels (analytic | numerical), rho only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "task_simulations" / "Shock_Tube"))

from sod_euler import EulerSolver1D        # noqa: E402
from sod_exact import exact_density_on_cells  # noqa: E402

# Default conditions ---------------------------------------------------------
TRUTH = np.array([1.0e5, 1.0e4, 1.0, 0.125])  # [p_high, p_low, rho_high, rho_low]
T_FINAL = 6.0e-4
N_CELLS = 256

p_high, p_low, rho_high, rho_low = TRUTH

# Analytic solution ----------------------------------------------------------
x = (np.arange(N_CELLS) + 0.5) / N_CELLS
rho_exact = exact_density_on_cells(TRUTH, T_FINAL, N_CELLS)

# Numerical solution ---------------------------------------------------------
solver = EulerSolver1D(nx=N_CELLS, xmin=0.0, xmax=1.0, gamma=1.4, cfl=0.5)
solver.set_sod_like(rho_high=rho_high, p_high=p_high, rho_low=rho_low, p_low=p_low, x0=0.5)
solver.step_to(T_FINAL)
rho_num = solver.primitive()[0]

# Plot -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

ax1.plot(x, rho_exact, color="C0", lw=1.6)
ax1.set_title("Analytic (exact Sod)")
ax1.set_xlabel("x")
ax1.set_ylabel("rho")
ax1.grid(alpha=0.3)

ax2.plot(x, rho_num, color="C0", lw=1.6)
ax2.set_title("Numerical (MUSCL-HLLC)")
ax2.set_xlabel("x")
ax2.grid(alpha=0.3)

fig.suptitle("Shock-tube density at p=(10000, 1000), rho=(1.0, 0.125), t=0.0006")
             
fig.tight_layout()

out = _ROOT / "figures" / "shock_tube_rho_exact_vs_numerical.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
