"""Shared infrastructure for the Dakota MCMC shock-tube workflow.

Everything under ``dakota_mcmc/`` imports from here. The problem constants,
observation builder and forward models come from the global
``helpers.inverse_common`` (shared with ``EnKF/`` and ``MCMC/``) and are
re-exported here; this module then adds only the Dakota-specific ensemble loader
and posterior figure. The only outside dependencies are that shared layer, the
physics stack in ``task_simulations/Shock_Tube/`` and the surrogate pickle for
``SOD_FORWARD=surrogate``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_simulations.Shock_Tube.sod_exact import exact_density_on_cells  # noqa: E402,F401
# Shared problem layout + observation model + forward models (single source of
# truth); re-exported so the dakota_mcmc scripts keep importing them from common.
from helpers.inverse_common import (  # noqa: E402,F401
    ENSEMBLE_DIR,
    FIELD_OFFSET,
    FORWARD_MODES,
    GAMMA,
    N_FIELD,
    N_PARAMS,
    N_STATE,
    OBS_ERROR,
    PARAM_NAMES,
    SURROGATE_PKL,
    T_FINAL,
    TRUTH,
    build_observations,
    euler_density,
    exact_density,
    load_ensemble,
    load_surrogate,
    make_forward,
    surrogate_density,
)


def make_plots(res: dict, outdir: Path, label: str = "dakota") -> None:
    """Marginal histograms, traces, posterior field band, and residuals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    post, chain = res["post"], res["chain"]
    burn = res["burn"]
    truth = res["truth"]
    tag = label.upper()

    fig, axes = plt.subplots(2, N_PARAMS, figsize=(16, 7), constrained_layout=True)
    for k, name in enumerate(PARAM_NAMES):
        ax = axes[0, k]
        ax.hist(post[:, k], bins=40, color="tab:blue", alpha=0.75, density=True)
        ax.axvline(truth[k], color="black", lw=2.0, label="truth")
        ax.axvline(res["mean"][k], color="tab:red", lw=1.6, ls="--", label="post. mean")
        ax.axvspan(res["quantiles"][0, k], res["quantiles"][2, k],
                   color="tab:red", alpha=0.12, label="95% CI")
        ax.set_title(name)
        ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")

        axt = axes[1, k]
        axt.plot(chain[:, k], color="0.4", lw=0.4)
        axt.axvline(burn, color="tab:orange", lw=1.2, ls=":", label="burn-in")
        axt.axhline(truth[k], color="black", lw=1.2)
        axt.set_xlabel("MCMC step")
        if k == 0:
            axt.set_ylabel("trace")
            axt.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{tag} posterior over Sod ICs  (g = {res['forward']}, "
                 f"acc = {res['acc_rate']:.2f})")
    p1 = outdir / f"{label}_marginals.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {p1}")

    x = res["x_cells"]
    fig2, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax.fill_between(x, res["field_lo"], res["field_hi"], color="tab:blue",
                    alpha=0.25, label="95% posterior band")
    ax.plot(x, res["field_mean"], color="tab:blue", lw=1.8,
            label=f"posterior mean (RMSE {res['post_rmse']:.2e})")
    ax.plot(x, res["exact"], color="black", lw=2.0, zorder=0, label="Exact")
    ax.scatter(res["obs_x"], res["obs_y"], color="gray", marker="o", s=16,
               edgecolor="white", linewidth=0.4, zorder=5, label="Observations")
    ax.set_ylabel("density rho")
    ax.set_title(f"{tag} reconstructed field  (g = {res['forward']}, t = {T_FINAL:.4f})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    axr.axhline(0.0, color="black", lw=0.8)
    axr.plot(x, res["field_mean"] - res["exact"], color="tab:blue", lw=1.4,
             label="posterior mean - exact")
    axr.scatter(res["obs_x"], np.zeros_like(res["obs_x"]), color="gray",
                marker="|", s=40, zorder=4)
    axr.set_xlabel("x")
    axr.set_ylabel("residual")
    axr.legend(loc="upper right", fontsize=9)
    axr.grid(True, alpha=0.3)
    p2 = outdir / f"{label}_field.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"[plot] wrote {p2}")
