"""Metropolis-Hastings MCMC for the 1D shock tube -- infer the 4 boundary
parameters from final-time observations.

This is the Bayesian counterpart to the point-estimate methods in
``inference/enkf/``: where ``enkf_driver.py`` does one Kalman update and
``es_mda.py`` does a few tempered ensemble updates, this samples the FULL
posterior over the 4 inflow / initial-condition parameters

    m = [p_high, p_low, rho_high, rho_low]

given observations of the density field at the final time ``T_FINAL``. It uses
the shared observation methodology and forward models from
``inference/common.py``, so the methods are directly comparable on the same
inverse problem.

Model
-----
* forward model  g(m) = Sod density at the observed cells at ``T_FINAL``
  (``--forward exact`` analytic Sod -- fast, but the same model that makes the
  observations, i.e. an inverse crime; ``--forward euler`` numerical solver --
  honest, slower; ``--forward surrogate`` trained GPR).
* data           d = exact Sod density at those cells for the TRUTH params,
  identical to ``common.build_observations``.
* likelihood     d | m ~ N(g(m), sigma^2 I),  sigma = ``--obs-error``.
* prior          independent Uniform on each parameter, bounds taken from the
  20-member ensemble's min/max padded by ``--prior-pad`` (so the truth is well
  inside and the support is comparable to ES-MDA's starting ensemble).

Sampler
-------
Random-walk Metropolis in a NORMALIZED parameter space u in [0, 1]^4 (so the
four very different scales -- p~1e5 vs rho~0.1 -- get comparable proposals).
Two proposal modes (``--proposal``):

* ``cov`` (default) -- Haario adaptive-covariance: the proposal covariance is the
  running sample covariance of the chain, C, scaled by ``--cov-scale`` (default
  2.38/sqrt(d)). This shapes the proposal to the posterior, which is strongly
  ANISOTROPIC here (rho is constrained ~20x tighter than p in normalized units),
  so a single isotropic step can't reach a good acceptance rate. A short
  isotropic warmup (``--cov-warmup``) seeds C before the adaptive proposal kicks in.
* ``iso`` -- a single isotropic step (the original behaviour).

In both modes a scalar multiplier is adapted toward 0.234 acceptance
(Robbins-Monro) during burn-in; after burn-in the kernel is frozen and samples
are kept (so the retained chain has a fixed transition kernel).

Run from the repo root::

    python inference/mcmc/mcmc_infer.py
    python inference/mcmc/mcmc_infer.py --steps 40000 --forward euler
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# This script lives two levels under the repo root (inference/mcmc/).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.common import (  # noqa: E402
    ENSEMBLE_DIR, FORWARD_MODES, N_FIELD, N_PARAMS, OBS_ERROR, PARAM_NAMES,
    T_FINAL, TRUTH,
    build_observations, load_ensemble, make_forward, prior_bounds,
)


# --------------------------------------------------------------------------- #
# Prior, likelihood, posterior (all in PHYSICAL parameter units).
# --------------------------------------------------------------------------- #
def log_posterior(m, g, d_obs, sigma, lo, hi) -> float:
    """log p(m | d) up to a constant; -inf outside the uniform prior box."""
    if np.any(m < lo) or np.any(m > hi):
        return -np.inf
    resid = g(m) - d_obs
    return -0.5 * np.sum(resid * resid) / (sigma * sigma)


# --------------------------------------------------------------------------- #
# Random-walk Metropolis in normalized [0, 1]^4 space.
# --------------------------------------------------------------------------- #
def run_mcmc(args) -> dict:
    rng = np.random.default_rng(args.seed)
    directory = Path(args.dir)

    # Operating point the observations are synthesized from (--truth overrides).
    truth = TRUTH if args.truth is None else np.asarray(args.truth, dtype=float)

    H, d_obs, cell_idx, exact_rho = build_observations(args.obs_every, args.margin,
                                                       truth=truth)
    g = make_forward(args.forward, cell_idx)
    lo, hi = prior_bounds(directory, args.prior_pad, contain=truth)
    span = hi - lo

    # Map physical params <-> normalized u in [0, 1]^4.
    to_u = lambda m: (m - lo) / span
    to_m = lambda u: lo + u * span
    logp = lambda m: log_posterior(m, g, d_obs, args.obs_error, lo, hi)

    print(f"[setup] MCMC steps={args.steps} burn={args.burn} thin={args.thin} "
          f"forward={args.forward} m={d_obs.size} obs_error={args.obs_error}")
    print(f"[truth] " + "  ".join(f"{n}={truth[k]:.4e}"
                                   for k, n in enumerate(PARAM_NAMES)))
    print("[prior] uniform bounds (pad={:.2f}):".format(args.prior_pad))
    for k, n in enumerate(PARAM_NAMES):
        print(f"    {n:9s} [{lo[k]:.4e}, {hi[k]:.4e}]")

    # Start at the ensemble-mean (center of the prior box is a fine alternative).
    u = to_u(load_ensemble(directory)[:N_PARAMS, :].mean(axis=1))
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    cur_lp = logp(to_m(u))

    d = N_PARAMS
    proposal = args.proposal
    # `scale` is the single Robbins-Monro-adapted multiplier (target acc 0.234).
    #   iso : proposal = u + scale * z                  (z ~ N(0, I))
    #   cov : proposal = u + scale * L z, L = chol(C)   (C = running posterior cov)
    scale = args.cov_scale if proposal == "cov" else args.step

    # Online moments of u (Welford) -> adaptive proposal covariance (Haario 2001).
    mu = u.copy()
    M2 = np.zeros((d, d))
    n_mom = 1
    chol = np.eye(d) * args.step           # seed until the cov warmup elapses

    def _cov_chol(C):
        """Cholesky of a regularized covariance; diagonal fallback if not PD."""
        reg = 1e-10 * np.trace(C) / d + 1e-12
        try:
            return np.linalg.cholesky(C + reg * np.eye(d))
        except np.linalg.LinAlgError:
            return np.diag(np.sqrt(np.clip(np.diag(C), 1e-12, None)))

    n_total = args.steps
    chain = np.empty((n_total, N_PARAMS))  # stored in PHYSICAL units
    lp_chain = np.empty(n_total)
    n_acc = 0
    acc_window = 0                         # acceptances since last adaptation

    for i in range(n_total):
        z = rng.standard_normal(d)
        use_cov = proposal == "cov" and i >= args.cov_warmup
        if use_cov:
            prop = u + scale * (chol @ z)
        else:
            # iso mode, or cov mode still seeding its covariance: small isotropic step.
            warm = scale if proposal == "iso" else args.step
            prop = u + warm * z
        prop_lp = logp(to_m(prop))
        if np.log(rng.random()) < prop_lp - cur_lp:
            u, cur_lp = prop, prop_lp
            n_acc += 1
            acc_window += 1

        chain[i] = to_m(u)
        lp_chain[i] = cur_lp

        # Update running mean/cov of the realized chain (every state, Welford).
        n_mom += 1
        delta = u - mu
        mu += delta / n_mom
        M2 += np.outer(delta, u - mu)

        # Adapt during burn-in only, so post-burn-in samples come from a fixed
        # kernel. Robbins-Monro on the scalar (target 0.234) + refresh the
        # proposal covariance from the accumulated moments.
        if args.adapt and i < args.burn and (i + 1) % args.adapt_every == 0:
            rate = acc_window / args.adapt_every
            if proposal == "iso" or use_cov:
                scale *= float(np.exp((rate - 0.234) / np.sqrt(i + 1.0)))
                lo_s, hi_s = (1e-3, 10.0) if proposal == "cov" else (1e-4, 1.0)
                scale = float(np.clip(scale, lo_s, hi_s))
            if proposal == "cov" and n_mom > d + 1:
                chol = _cov_chol(M2 / (n_mom - 1))
            acc_window = 0

    # Posterior = post-burn-in, thinned.
    post = chain[args.burn::args.thin]
    acc_rate = n_acc / n_total

    mean = post.mean(axis=0)
    std = post.std(axis=0)
    q = np.percentile(post, [2.5, 50, 97.5], axis=0)
    map_idx = int(np.argmax(lp_chain))
    map_params = chain[map_idx]

    print(f"[run] proposal={proposal}  acceptance={acc_rate:.3f}  "
          f"final scale={scale:.4f}  posterior samples={post.shape[0]}")
    print("[posterior] mean +/- std   [2.5%, 97.5%]   (truth):")
    for k, n in enumerate(PARAM_NAMES):
        print(f"    {n:9s} {mean[k]:.4e} +/- {std[k]:.2e}   "
              f"[{q[0, k]:.4e}, {q[2, k]:.4e}]   (truth {truth[k]:.4e})")

    # Reconstruct full density field from posterior draws -> credible band.
    g_full = make_forward(args.forward)
    draw_idx = rng.choice(post.shape[0], size=min(args.n_field_draws, post.shape[0]),
                          replace=False)
    fields = np.array([g_full(post[j]) for j in draw_idx])    # (Ndraw x N_FIELD)
    field_mean = fields.mean(axis=0)
    field_lo, field_hi = np.percentile(fields, [2.5, 97.5], axis=0)
    post_rmse = float(np.sqrt(np.mean((field_mean - exact_rho) ** 2)))
    print(f"[result] posterior-mean density RMSE vs exact = {post_rmse:.4e}")

    return {
        "chain": chain, "lp_chain": lp_chain, "post": post, "truth": truth,
        "mean": mean, "std": std, "quantiles": q,
        "map_params": map_params, "acc_rate": acc_rate,
        "x_cells": np.arange(N_FIELD) / N_FIELD,
        "exact": exact_rho, "field_mean": field_mean,
        "field_lo": field_lo, "field_hi": field_hi,
        "obs_x": cell_idx / N_FIELD, "obs_y": d_obs,
        "post_rmse": post_rmse, "lo": lo, "hi": hi,
        "burn": args.burn, "forward": args.forward,
    }


# --------------------------------------------------------------------------- #
# Plots: marginal histograms, traces, and the reconstructed field band.
# --------------------------------------------------------------------------- #
def make_plots(res: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    post, chain = res["post"], res["chain"]
    burn = res["burn"]
    truth = res["truth"]

    # --- marginals + traces (2 rows x 4 cols) -------------------------------
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
    fig.suptitle(f"MCMC posterior over Sod ICs  (g = {res['forward']}, "
                 f"acc = {res['acc_rate']:.2f})")
    p1 = outdir / "mcmc_marginals.png"
    fig.savefig(p1, dpi=150)
    print(f"[plot] wrote {p1}")

    # --- reconstructed density field with 95% band -------------------------
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
    ax.set_title(f"MCMC reconstructed field  (g = {res['forward']}, t = {T_FINAL:.4f})")
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
    p2 = outdir / "mcmc_field.png"
    fig2.savefig(p2, dpi=150)
    print(f"[plot] wrote {p2}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ENSEMBLE_DIR),
                    help="directory of the 20 ensemble HDF5 files (for prior bounds)")
    ap.add_argument("--forward", choices=FORWARD_MODES, default="exact",
                    help="forward model g(m): 'exact' analytic Sod (inverse crime, fast), "
                         "'euler' numerical solver (honest, slower), or 'surrogate' "
                         "trained GPR")
    ap.add_argument("--truth", type=float, nargs=4, default=None,
                    metavar=("P_HIGH", "P_LOW", "RHO_HIGH", "RHO_LOW"),
                    help="operating point the observations are synthesized from "
                         "(default: common.TRUTH = 1e5 1e4 1.0 0.125). The prior "
                         "box auto-expands to contain it.")
    ap.add_argument("--steps", type=int, default=20000, help="total MCMC steps")
    ap.add_argument("--burn", type=int, default=4000, help="burn-in steps to discard")
    ap.add_argument("--thin", type=int, default=5, help="keep every k-th post-burn sample")
    ap.add_argument("--proposal", choices=["cov", "iso"], default="cov",
                    help="'cov': Haario adaptive-covariance proposal that learns the "
                         "posterior shape (handles the rho/p anisotropy); 'iso': single "
                         "isotropic step")
    ap.add_argument("--step", type=float, default=0.05,
                    help="isotropic proposal std in normalized [0,1] space ('iso' mode, "
                         "and the seeding step during the 'cov' warmup)")
    ap.add_argument("--cov-scale", dest="cov_scale", type=float, default=2.38 / np.sqrt(N_PARAMS),
                    help="initial scalar multiplier on chol(C) in 'cov' mode "
                         "(2.38/sqrt(d) is the Roberts-Rosenthal optimum)")
    ap.add_argument("--cov-warmup", dest="cov_warmup", type=int, default=300,
                    help="'cov' mode: isotropic steps for this many iters to seed the "
                         "covariance before switching to the adaptive proposal")
    ap.add_argument("--no-adapt", dest="adapt", action="store_false",
                    help="disable Robbins-Monro step adaptation during burn-in")
    ap.add_argument("--adapt-every", dest="adapt_every", type=int, default=100,
                    help="adapt the step size every k steps during burn-in")
    ap.add_argument("--prior-pad", dest="prior_pad", type=float, default=0.25,
                    help="pad the ensemble min/max prior box by this fraction of its half-width")
    ap.add_argument("--n-field-draws", dest="n_field_draws", type=int, default=300,
                    help="posterior draws used to build the reconstructed-field band")
    # Observation methodology -- identical flags to enkf_driver / es_mda.
    ap.add_argument("--obs-every", type=int, default=15,
                    help="baseline: observe every k-th density cell in the flat regions")
    ap.add_argument("--margin", type=int, default=4,
                    help="exclude cells within this many of each discontinuity")
    ap.add_argument("--obs-error", type=float, default=OBS_ERROR,
                    help="observation noise std (likelihood sigma)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    args = ap.parse_args()

    res = run_mcmc(args)
    if not args.no_plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        make_plots(res, outdir)


if __name__ == "__main__":
    main()
