"""Quantify and visualize uncertain / abnormal predictions of a trained
shock-tube rho surrogate.

The surrogate (``task_train/model.py::Surrogate``, ``pod_inc=False``) maps the
five inputs ``[p_high, p_low, rho_high, rho_low, t]`` to a 256-point density
profile ``rho(x)``. It is accurate over most of the design box but some
parameter / time combinations produce "weird" profiles (smeared or ringing
discontinuities). This script makes that visible three ways:

  1. **Intrinsic uncertainty** - the GP's own predictive std (``return_std``),
     swept over time and over a 2-D parameter slice, with training samples
     overlaid so under-sampled regions are obvious.
  2. **Validated error** - the surrogate is compared against the *real* Euler
     solver on a fresh random sample, confirming that high GP std and late time
     coincide with high true error.
  3. **Failure gallery** - the worst predicted profiles drawn on top of ground
     truth, with the physical-abnormality regions (over/undershoot) shaded.

Run:  python analysis/surrogate_uncertainty.py
Figures land in analysis/figures/.
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
import types
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.parameter_spaces import get_parameter_space  # noqa: E402
from task_simulations.Shock_Tube.sod_euler import EulerSolver1D  # noqa: E402

RUN_DIR = ROOT / "training_runs" / "shock_tube" / "run_200" / "wf_0"
FIG_DIR = Path(__file__).resolve().parent / "figures"
PARAM_LABELS = ["p_high", "p_low", "rho_high", "rho_low"]


def load_surrogate(path: Path):
    """Unpickle a surrogate, aliasing the legacy ``surrogate.*`` module names."""
    import task_train.model as model_mod
    import task_train.POD as pod_mod

    pkg = types.ModuleType("surrogate")
    pkg.__path__ = []  # mark as package
    sys.modules.setdefault("surrogate", pkg)
    sys.modules.setdefault("surrogate.model", model_mod)
    sys.modules.setdefault("surrogate.POD", pod_mod)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def spatial_and_time_grid(run_dir: Path):
    f = sorted(glob.glob(str(run_dir.parent / "data" / "*.h5")))[0]
    with h5py.File(f, "r") as h:
        return np.asarray(h["x"], float), np.asarray(h["t"], float)


def true_rho(params, t_query) -> np.ndarray:
    """Ground-truth density at ``t_query`` from the real 1-D Euler solver."""
    solver = EulerSolver1D(nx=256, xmin=0.0, xmax=1.0, gamma=1.4, cfl=0.5)
    solver.set_sod_like(
        rho_high=params[2], p_high=params[0], rho_low=params[3], p_low=params[1], x0=0.5
    )
    solver.step_to(float(t_query))
    return solver.primitive()[0]


# ---------------------------------------------------------------------------
# Figure 1 - intrinsic GP uncertainty
# ---------------------------------------------------------------------------
def fig_uncertainty(surr, lo, hi, t_grid, train_phys):
    rng = np.random.default_rng(0)

    # (A) uncertainty vs time, over many random parameter draws. The 61 training
    # time-nodes (spacing 1e-5) are all sampled, so a fine sweep reveals a "comb":
    # the GP std collapses at trained slices and rises in the gaps between them.
    n = 300
    P = lo + (hi - lo) * rng.random((n, 4))
    times = np.linspace(t_grid[0], t_grid[-1], 1200)
    std_t = np.empty((len(times), n))
    for i, t in enumerate(times):
        X = np.column_stack([P, np.full(n, t)])
        _, std = surr.predict(X, return_std=True)
        std_t[i] = std

    # (B) std heatmap over (p_high, p_low) at the final time, others at midbox
    ng = 60
    g_ph = np.linspace(lo[0], hi[0], ng)
    g_pl = np.linspace(lo[1], hi[1], ng)
    GPh, GPl = np.meshgrid(g_ph, g_pl)
    mid_rho_hi = 0.5 * (lo[2] + hi[2])
    mid_rho_lo = 0.5 * (lo[3] + hi[3])
    t_final = t_grid[-1]
    Xg = np.column_stack(
        [GPh.ravel(), GPl.ravel(),
         np.full(GPh.size, mid_rho_hi), np.full(GPh.size, mid_rho_lo),
         np.full(GPh.size, t_final)]
    )
    _, std_g = surr.predict(Xg, return_std=True)
    std_g = std_g.reshape(GPh.shape)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    med = np.median(std_t, axis=1)
    ax[0].fill_between(times, np.percentile(std_t, 10, axis=1),
                       np.percentile(std_t, 90, axis=1), alpha=0.25,
                       color="C0", label="10-90th pct")
    ax[0].plot(times, med, "C0-", lw=0.8, label="median")
    for tn in t_grid:
        ax[0].axvline(tn, color="grey", lw=0.3, alpha=0.4)
    ax[0].set_xlabel("simulation time  t  [s]")
    ax[0].set_ylabel("GP predictive std  (scaled rho units)")
    ax[0].set_title("Uncertainty collapses at trained time slices,\n"
                    "spikes in the gaps between them (grey = training nodes)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    # zoom inset on a single inter-node gap to show the dip-and-rise clearly
    axin = ax[0].inset_axes([0.55, 0.55, 0.42, 0.4])
    m = (times >= 2.9e-4) & (times <= 3.3e-4)
    axin.plot(times[m], med[m], "C0-", lw=1.0)
    for tn in t_grid:
        if 2.9e-4 <= tn <= 3.3e-4:
            axin.axvline(tn, color="grey", lw=0.6, alpha=0.6)
    axin.set_title("zoom: one gap", fontsize=8)
    axin.tick_params(labelsize=6)

    pcm = ax[1].pcolormesh(GPh, GPl, std_g, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax[1], label="GP predictive std")
    # training samples taken at (or near) the final time, projected onto slice
    near_final = train_phys[np.abs(train_phys[:, 4] - t_final) < 1e-6]
    ax[1].scatter(near_final[:, 0], near_final[:, 1], s=14, c="white",
                  edgecolors="k", linewidths=0.4, label="training samples")
    ax[1].set_xlabel("p_high")
    ax[1].set_ylabel("p_low")
    ax[1].set_title(f"Uncertainty over (p_high, p_low) @ t={t_final:.1e}\n"
                    "(rho_high, rho_low at mid-box)")
    ax[1].legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = FIG_DIR / "1_uncertainty.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 1b - uncertainty heatmaps for both parameter pairs, side by side
# ---------------------------------------------------------------------------
def _std_slice(surr, lo, hi, i, j, fixed, t_final, ng=60):
    """GP std over a 2-D slice varying params i, j with the rest at ``fixed``."""
    gi = np.linspace(lo[i], hi[i], ng)
    gj = np.linspace(lo[j], hi[j], ng)
    Gi, Gj = np.meshgrid(gi, gj)
    P = np.tile(fixed, (Gi.size, 1)).astype(float)
    P[:, i] = Gi.ravel()
    P[:, j] = Gj.ravel()
    X = np.column_stack([P, np.full(Gi.size, t_final)])
    _, std = surr.predict(X, return_std=True)
    return Gi, Gj, std.reshape(Gi.shape)


def fig_uncertainty_pairs(surr, lo, hi, t_grid, train_phys):
    """Two GP-std heatmaps at the final time: pressures and densities."""
    t_final = t_grid[-1]
    # The actual training data spills past the declared design bounds (e.g.
    # rho_low reaches ~0.094 vs a declared 0.115), so grid over the real data
    # extent — otherwise the heatmap leaves an uncolored strip under those points.
    lo = np.minimum(lo, train_phys[:, :4].min(axis=0))
    hi = np.maximum(hi, train_phys[:, :4].max(axis=0))
    mid = 0.5 * (lo + hi)

    # (left) vary pressures (0,1), densities at mid-box
    Gi0, Gj0, S0 = _std_slice(surr, lo, hi, 0, 1, mid, t_final)
    # (right) vary densities (2,3), pressures at mid-box
    Gi1, Gj1, S1 = _std_slice(surr, lo, hi, 2, 3, mid, t_final)

    # shared color scale so the two panels are directly comparable
    vmin = min(S0.min(), S1.min())
    vmax = max(S0.max(), S1.max())

    near_final = train_phys[np.abs(train_phys[:, 4] - t_final) < 1e-6]

    def fmt(k):  # held-parameter value, formatted by magnitude
        return f"{mid[k]:.0f}" if mid[k] >= 100 else f"{mid[k]:.3f}"

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    specs = [
        (ax[0], Gi0, Gj0, S0, 0, 1, "p_high", "p_low",
         f"(rho_high={fmt(2)}, rho_low={fmt(3)})"),
        (ax[1], Gi1, Gj1, S1, 2, 3, "rho_high", "rho_low",
         f"(p_high={fmt(0)}, p_low={fmt(1)})"),
    ]
    pcm = None
    for a, Gi, Gj, S, i, j, xl, yl, sub in specs:
        pcm = a.pcolormesh(Gi, Gj, S, shading="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax)
        a.scatter(near_final[:, i], near_final[:, j], s=14, c="white",
                  edgecolors="k", linewidths=0.4, label="training samples")
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.set_title(f"Uncertainty over ({xl}, {yl}) at t={t_final:.1e}\n{sub}")
        a.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(pcm, ax=ax, label="GP predictive std (shared scale)",
                 fraction=0.046, pad=0.04)

    out = FIG_DIR / "1b_uncertainty_pairs.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figures 2 & 3 - validated error and worst-case gallery
# ---------------------------------------------------------------------------
def validate(surr, lo, hi, t_grid, n_params=80):
    rng = np.random.default_rng(1)
    times = np.array([6e-5, 2e-4, 4e-4, 6e-4])
    recs = []
    for _ in range(n_params):
        P = lo + (hi - lo) * rng.random(4)
        for t in times:
            yt = true_rho(P, t)
            X = np.append(P, t)[None, :]
            yp, std = surr.predict(X, return_std=True)
            yp = yp[0]
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            # physical abnormality: excursions outside the [rho_low, rho_high] band
            over = float(np.max(yp - P[2]).clip(min=0))   # above the high density
            under = float((P[3] - np.min(yp)).clip(min=0))  # below the low density
            recs.append(dict(rmse=rmse, std=float(std[0]), t=float(t),
                             P=P.copy(), yp=yp, yt=yt,
                             over=over, under=under))
    return recs


def fig_calibration(surr, recs):
    """Compare the surrogate's self-reported uncertainty (GP std) against its
    actual error, both in the *same* scaled-rho units the GP std lives in.

    Perfect calibration would put points on the y=x line: predicted std equal to
    the realized RMSE. Points above the line mean the surrogate is *over-confident*
    (real error exceeds what it claims).
    """
    std = np.array([r["std"] for r in recs])           # scaled-rho units
    tt = np.array([r["t"] for r in recs])
    # realized error in the SAME scaled space as the GP std
    err = np.array([
        float(np.sqrt(np.mean(
            (surr.y_scaler.transform(r["yp"][None, :])
             - surr.y_scaler.transform(r["yt"][None, :])) ** 2)))
        for r in recs
    ])

    lo_l = min(std.min(), err.min()) * 0.7
    hi_l = max(std.max(), err.max()) * 1.4

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # (left) per-prediction scatter on log-log axes (claimed std and actual error
    # span very different magnitudes, so linear axes squash everything to a line)
    sc = ax[0].scatter(std, err, c=tt, cmap="plasma", s=24, alpha=0.85)
    fig.colorbar(sc, ax=ax[0], label="time t [s]")
    ax[0].plot([lo_l, hi_l], [lo_l, hi_l], "k--", lw=1,
               label="perfect calibration (y=x)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim(lo_l, hi_l)
    ax[0].set_ylim(lo_l, hi_l)
    ax[0].set_xlabel("GP predictive std  (what the surrogate claims)")
    ax[0].set_ylabel("actual RMSE vs Euler  (scaled rho units)")
    ax[0].set_title("Claimed uncertainty vs realized error (log-log)\n"
                    "points above y=x = over-confident")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3, which="both")

    # (right) calibration curve: bin by claimed std, show mean realized error.
    # Zoomed to the actual std range so the (nearly flat) claimed-std axis is
    # readable — the curve being far above y=x is the whole point.
    nb = 8
    edges = np.quantile(std, np.linspace(0, 1, nb + 1))
    edges[-1] += 1e-12
    idx = np.clip(np.digitize(std, edges) - 1, 0, nb - 1)
    bx, by, bs = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() < 2:
            continue
        bx.append(std[m].mean())
        by.append(err[m].mean())
        bs.append(err[m].std())
    bx, by, bs = map(np.array, (bx, by, bs))
    xr = (std.min() * 0.9, std.max() * 1.05)
    ax[1].plot(xr, xr, "k--", lw=1, label="perfect calibration (y=x)")
    ax[1].errorbar(bx, by, yerr=bs, fmt="o-", color="C0", capsize=3,
                   label="binned mean ± std")
    ax[1].set_xlim(*xr)
    ax[1].set_ylim(0, max(by + bs) * 1.1)
    ax[1].set_xlabel("mean GP predictive std in bin")
    ax[1].set_ylabel("mean actual RMSE in bin")
    ratio = np.median(err / np.maximum(std, 1e-12))
    ax[1].set_title("Reliability curve (8 std-quantile bins)\n"
                    f"median actual/claimed ratio = {ratio:.1f}x")
    ax[1].legend(loc="upper left")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "4_calibration.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, ratio


def fig_error(recs):
    rmse = np.array([r["rmse"] for r in recs])
    std = np.array([r["std"] for r in recs])
    tt = np.array([r["t"] for r in recs])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    sc = ax[0].scatter(std, rmse, c=tt, cmap="plasma", s=22, alpha=0.85)
    fig.colorbar(sc, ax=ax[0], label="time t [s]")
    cc = np.corrcoef(std, rmse)[0, 1]
    ax[0].set_xlabel("GP predictive std (self-reported uncertainty)")
    ax[0].set_ylabel("true RMSE vs Euler solver")
    ax[0].set_title("Does the surrogate know when it's wrong?\n"
                    f"only weakly: Pearson r = {cc:.2f}")
    ax[0].grid(alpha=0.3)

    # RMSE distribution per time
    times = sorted(set(tt))
    data = [rmse[tt == t] for t in times]
    ax[1].boxplot(data, labels=[f"{t:.0e}" for t in times], showfliers=True)
    ax[1].set_xlabel("simulation time t [s]")
    ax[1].set_ylabel("true RMSE")
    ax[1].set_title("Error concentrates at late time\n(sharp shock/contact structure)")
    ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = FIG_DIR / "2_error_validation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, cc


def fig_gallery(recs, x):
    worst = sorted(recs, key=lambda r: r["rmse"], reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, r in zip(axes.ravel(), worst):
        P = r["P"]
        ax.plot(x, r["yt"], "k-", lw=1.8, label="truth (Euler)")
        ax.plot(x, r["yp"], "C3--", lw=1.8, label="surrogate")
        # shade physical-abnormality band [rho_low, rho_high]
        ax.axhspan(P[3], P[2], color="green", alpha=0.05)
        ax.axhline(P[2], color="green", ls=":", lw=0.8)
        ax.axhline(P[3], color="green", ls=":", lw=0.8)
        # highlight where the surrogate leaves the physical band
        bad = (r["yp"] > P[2] + 1e-6) | (r["yp"] < P[3] - 1e-6)
        ax.scatter(x[bad], r["yp"][bad], s=10, c="orange", zorder=5,
                   label="out-of-band" if bad.any() else None)
        title = (f"p_hi={P[0]:.0f} p_lo={P[1]:.0f} "
                 f"rho={P[2]:.2f}/{P[3]:.2f} t={r['t']:.0e}\n"
                 f"RMSE={r['rmse']:.4f}  GPstd={r['std']:.4f}  "
                 f"over={r['over']:.3f} under={r['under']:.3f}")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("rho")
    fig.suptitle("Worst predicted density profiles: smeared / ringing "
                 "discontinuities vs ground truth", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "3_worst_profiles.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    FIG_DIR.mkdir(exist_ok=True)
    ps = get_parameter_space("shock_tube")
    lo = np.array(ps.l_bounds, float)
    hi = np.array(ps.u_bounds, float)

    surr = load_surrogate(RUN_DIR / "surrogate.pkl")
    x, t_grid = spatial_and_time_grid(RUN_DIR)
    train_phys = surr.x_scaler.inverse_transform(surr.gp.X_train_)

    f1 = fig_uncertainty(surr, lo, hi, t_grid, train_phys)
    f1b = fig_uncertainty_pairs(surr, lo, hi, t_grid, train_phys)
    recs = validate(surr, lo, hi, t_grid)
    f2, cc = fig_error(recs)
    f3 = fig_gallery(recs, x)
    f4, ratio = fig_calibration(surr, recs)

    rmse = np.array([r["rmse"] for r in recs])
    print("Saved figures:")
    for p in (f1, f1b, f2, f3, f4):
        print("  ", p.relative_to(ROOT))
    print(f"\nValidation summary ({len(recs)} param/time evaluations):")
    print(f"  RMSE  median={np.median(rmse):.4f}  "
          f"p90={np.percentile(rmse, 90):.4f}  max={rmse.max():.4f}")
    print(f"  corr(GP std, true RMSE) = {cc:.2f}")
    print(f"  median actual/claimed error ratio = {ratio:.1f}x")


if __name__ == "__main__":
    main()
