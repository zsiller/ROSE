"""Plotting helpers for Dakota MCMC validation (surrogate pushforward + PIT)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pit_values(pred: np.ndarray, obs: np.ndarray, obs_error: float,
               rng: np.random.Generator) -> np.ndarray:
    samples = pred
    if obs_error > 0.0:
        samples = pred + rng.normal(0.0, obs_error, size=pred.shape)
    return (samples <= obs[None, :]).mean(axis=0)


def uniformity_ks(pit: np.ndarray) -> str:
    try:
        from scipy import stats
        ks_stat, ks_p = stats.kstest(pit, "uniform")
        return f"KS={ks_stat:.3f}  p={ks_p:.3f}"
    except Exception:
        m = pit.size
        ecdf = np.arange(1, m + 1) / m
        ks_stat = float(np.max(np.abs(ecdf - np.sort(pit))))
        return f"KS={ks_stat:.3f}  p=n/a"


def plot_field_violins(out: Path, *, obs_x: np.ndarray, pred: np.ndarray,
                       obs: np.ndarray, pmed: np.ndarray, exact_rho: np.ndarray,
                       n_field: int, obs_every: int, t: float, n_draws: int) -> None:
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})

    width = 0.9 / n_field * obs_every
    parts = ax.violinplot([pred[:, j] for j in range(obs_x.size)],
                          positions=obs_x, widths=width, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.45)

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


def plot_pit(out: Path, *, pit: np.ndarray, ks_str: str, chain_name: str) -> None:
    m = pit.size
    fig, (axh, axp) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    nbins = max(4, min(10, m // 2))
    axh.hist(pit, bins=nbins, range=(0, 1), color="tab:blue", alpha=0.75,
             edgecolor="white")
    axh.axhline(m / nbins, color="black", lw=1.6, ls="--",
                label=f"uniform ({m}/{nbins} per bin)")
    axh.set_xlabel("PIT  $F_{pred}(y_{obs})$")
    axh.set_ylabel("count")
    axh.set_title(f"PIT histogram ({m} obs points)\nU-shape=over-confident, "
                  f"hump=under-confident, slope=biased")
    axh.legend(fontsize=9)

    srt = np.sort(pit)
    unif_q = (np.arange(1, m + 1) - 0.5) / m
    band = 1.358 / np.sqrt(m)
    axp.plot([0, 1], [0, 1], color="black", lw=1.4, ls="--", label="ideal (uniform)")
    axp.plot(unif_q, srt, "o-", color="tab:blue", ms=5, label="empirical PIT")
    axp.fill_between([0, 1], [-band, 1 - band], [band, 1 + band],
                     color="gray", alpha=0.15, label="95% KS band")
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1)
    axp.set_xlabel("uniform quantile")
    axp.set_ylabel("sorted PIT")
    axp.set_title(f"PIT PP-plot   ({ks_str})")
    axp.legend(fontsize=9, loc="upper left")
    axp.grid(True, alpha=0.3)

    fig.suptitle(f"Surrogate pushforward calibration vs observations  "
                 f"(chain: {chain_name})")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")
