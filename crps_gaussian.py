#!/usr/bin/env python
"""Gaussian predictive CDF vs. a scalar observation for CRPS illustration.

    CRPS(F, y) = integral ( F(x) - 1{x >= y} )^2 dx

Run from this directory::

    python crps_gaussian_illustration.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
OUT = HERE / "crps_illustration.png"

MU = 1.0
SIGMA = 1.0
Y_OBS = 0.0


def crps_gaussian(mu: float, sigma: float, y: float, grid: np.ndarray) -> float:
    F = norm.cdf(grid, loc=mu, scale=sigma)
    H = (grid >= y).astype(float)
    return float(np.trapezoid((F - H) ** 2, grid))


if __name__ == "__main__":
    lo = min(MU - 4 * SIGMA, Y_OBS)
    hi = max(MU + 4 * SIGMA, Y_OBS)
    grid = np.linspace(lo, hi, 4000)

    F = norm.cdf(grid, loc=MU, scale=SIGMA)
    H = (grid >= Y_OBS).astype(float)
    crps = crps_gaussian(MU, SIGMA, Y_OBS, grid)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.fill_between(grid, F, H, color="tab:orange", alpha=0.25)
    ax.plot(grid, F, color="tab:blue", lw=2.0,
            label=rf"predictive CDF $\mathcal{{N}}({MU:.1f},\,{SIGMA:.1f}^2)$")
    ax.plot(grid, H, color="crimson", lw=2.0,
            label=rf"observation CDF ($y'={Y_OBS:.1f}$)")
    ax.axvline(Y_OBS, color="crimson", lw=1.0, ls=":")
    ax.axvline(MU, color="tab:blue", lw=1.2, ls="--",
               label=rf"predictive mean $\mu={MU:.1f}$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("cumulative probability")
    ax.set_ylim(-0.06, 1.06)
    ax.set_title(
        f"   (CRPS $={crps:.3f}$)")
    ax.legend(loc="center right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {OUT}")
    print(f"[crps] N({MU}, {SIGMA}^2) vs y'={Y_OBS}: CRPS={crps:.4f}")
