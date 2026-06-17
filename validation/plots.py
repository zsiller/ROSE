#!/usr/bin/env python
"""Plotting for the surrogate-vs-observations validation (see ``val.py``).

Kept separate from the compute path so ``val.py`` only assembles the predictive
arrays and these functions own all the matplotlib. Each takes already-computed
arrays + an output path and writes one figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_field_violins(out: Path, *, obs_x: np.ndarray, pred: np.ndarray,
                       obs: np.ndarray, pmed: np.ndarray, exact_rho: np.ndarray,
                       n_field: int, obs_every: int, t: float, n_draws: int) -> None:
    """One violin per observation point at its x-location, with the observed
    value, surrogate median, and exact Sod field overlaid (+ residual panel)."""
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})

    width = 0.9 / n_field * obs_every
    parts = ax.violinplot([pred[:, j] for j in range(obs_x.size)],
                          positions=obs_x, widths=width, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("tab:blue"); body.set_alpha(0.45)

    ax.plot(np.arange(n_field) / n_field, exact_rho, color="black", lw=1.6,
            zorder=1, label="exact Sod field")
    ax.scatter(obs_x, obs, color="crimson", marker="o", s=34, zorder=5,
               edgecolor="white", linewidth=0.5, label="observed")
    ax.scatter(obs_x, pmed, color="tab:blue", marker="_", s=120, zorder=6,
               label="surrogate median")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title(f"Surrogate prediction distribution at each observation point "
                 f"({n_draws} chain draws, t = {t:.4e})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    axr.axhline(0.0, color="black", lw=0.8)
    axr.vlines(obs_x, np.zeros_like(obs_x), pmed - obs, color="tab:blue", lw=2.0)
    axr.scatter(obs_x, pmed - obs, color="tab:blue", s=20, zorder=5)
    axr.set_xlabel("x")
    axr.set_ylabel("median − obs")
    axr.grid(True, alpha=0.3)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_obs_histograms(out: Path, *, cell_idx: np.ndarray, obs_x: np.ndarray,
                        pred: np.ndarray, obs: np.ndarray, pmed: np.ndarray,
                        plo: np.ndarray, phi: np.ndarray, t: float,
                        n_draws: int) -> None:
    """4-column grid of per-observation-point histograms of the predictions."""
    m = cell_idx.size
    ncol = 4
    nrow = int(np.ceil(m / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             constrained_layout=True)
    for j, ax in enumerate(axes.ravel()):
        if j >= m:
            ax.axis("off")
            continue
        ax.hist(pred[:, j], bins=30, color="tab:blue", alpha=0.75, density=True)
        ax.axvline(obs[j], color="crimson", lw=2.0, label="observed")
        ax.axvline(pmed[j], color="tab:blue", lw=1.6, ls="--", label="median")
        ax.axvspan(plo[j], phi[j], color="tab:blue", alpha=0.12, label="95% band")
        ax.set_title(f"cell {cell_idx[j]}  (x={obs_x[j]:.3f})", fontsize=10)
        ax.set_yticks([])
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Surrogate prediction distribution per observation point "
                 f"({n_draws} chain draws, t = {t:.4e})")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_crps(out: Path, *, obs_x: np.ndarray, cell_idx: np.ndarray,
              crps: np.ndarray, exact_rho: np.ndarray, pmed: np.ndarray,
              fmed: np.ndarray, flo: np.ndarray, fhi: np.ndarray,
              obs: np.ndarray, n_field: int,
              t: float, chain_name: str) -> None:
    """Per-observation-point CRPS lined up under the solution it scores.

    Top panel: exact Sod field + the surrogate predictive median / 95% band
    over the *full domain*, with blue dots at the observation cells (surrogate
    median there) and red dots for the observed values. Bottom panel: per-cell
    CRPS (lower = better) at the same x-locations on a shared x-axis, so spatial
    CRPS spikes line up with the solution features (contact / shock) that drive
    them."""
    fig, (axs, axc) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 2]})

    # Top: full-domain solution overlay.
    x_full = np.arange(n_field) / n_field
    axs.plot(x_full, exact_rho, color="black", lw=1.6, zorder=1,
             label="exact Sod field (chain median)")
    axs.plot(x_full, fmed, color="tab:blue", lw=1.4, zorder=2,
             label="surrogate median")
    axs.fill_between(x_full, flo, fhi, color="tab:blue", alpha=0.2, zorder=0,
                     label="95% predictive band")
    axs.scatter(obs_x, pmed, color="tab:blue", marker="o", s=30, zorder=5,
                edgecolor="white", linewidth=0.5, label="surrogate @ obs cell")
    axs.scatter(obs_x, obs, color="crimson", marker="o", s=34, zorder=6,
                edgecolor="white", linewidth=0.5, label="observed")
    axs.set_ylabel(r"density $\rho$")
    axs.set_title(f"Surrogate prediction vs exact Sod over the domain "
                  f"(t = {t:.4e})")
    axs.legend(fontsize=9, loc="upper right", framealpha=0.95)
    axs.grid(True, alpha=0.3)

    # Bottom: CRPS bars on the shared x-axis.
    width = 0.8 * (np.median(np.diff(obs_x)) if obs_x.size > 1 else 0.02)
    axc.bar(obs_x, crps, width=width, color="tab:blue", alpha=0.75,
            edgecolor="white", label="per-cell CRPS")
    axc.axhline(crps.mean(), color="crimson", lw=1.6, ls="--",
                label=f"mean CRPS = {crps.mean():.3e}")
    for x, c, idx in zip(obs_x, crps, cell_idx):
        axc.annotate(str(int(idx)), (x, c), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=7)
    axc.set_xlabel("x")
    axc.set_ylabel("CRPS (density units)")
    axc.set_title(f"Per-cell CRPS ({crps.size} obs points, lower is better)")
    axc.legend(fontsize=9, loc="upper right")
    axc.grid(True, alpha=0.3)

    fig.suptitle(f"Surrogate pushforward CRPS vs observations  "
                 f"(chain: {chain_name})")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")
