"""Vanilla random-walk Metropolis for the 1D shock tube -- infer the 4 Sod
initial-condition parameters from final-time observations.

This is the DELIBERATELY UNADAPTED baseline next to ``MCMC/mh_mcmc.py``: a plain
Metropolis-Hastings sampler with a FIXED isotropic Gaussian proposal. No Haario
adaptive covariance, no Robbins-Monro step tuning, no warmup-phase kernel
changes -- the proposal you set on the command line is the proposal used for
every step, start to finish. The point is to see how a textbook RW-Metropolis
performs on this strongly anisotropic posterior (rho is constrained ~20x tighter
than p in normalized units) WITHOUT any of the machinery that hides that
difficulty.

Everything physical -- forward models, observation methodology, prior box, the
posterior plots -- is shared with the other samplers via ``MCMC/mcmc_common.py``.

Model
-----
* forward model  g(m) = Sod density at the observed cells at ``T_FINAL``
  (``--forward exact`` analytic Sod -- fast, inverse crime; ``--forward euler``
  numerical solver -- honest, slower; ``--forward surrogate`` trained GPR).
* data           d = exact Sod density at those cells for the TRUTH params.
* likelihood     d | m ~ N(g(m), sigma^2 I),  sigma = ``--obs-error``.
* prior          independent Uniform on each parameter, ``center * (1 -/+ frac)``
                 (``--prior-frac``), centered on the truth.

Sampler
-------
Random-walk Metropolis in a NORMALIZED parameter space u in [0, 1]^4 (so the
four very different scales -- p~1e5 vs rho~0.1 -- get a comparable proposal).
A single fixed isotropic step is used throughout::

    u' = u + step * z,   z ~ N(0, I_4)

with ``step`` the std in normalized units. Because nothing adapts, ``--step`` is
the one knob that matters: too large and acceptance collapses on the tight rho
directions; too small and the chain crawls along the loose p directions. Sweep
it to map the trade-off the adaptive sampler papers over.

Optional covariance proposals (no longer strictly 'vanilla', for comparison):

* ``--ensemble-dir DIR`` -- seed a FIXED proposal covariance from the HDF5
  ensemble's parameter covariance (a preconditioner). NB this is the PRIOR
  covariance, which is wider than the posterior, so the 2.38/sqrt(d) default
  ``--cov-scale`` over-disperses (near-zero acceptance) -- lower it (e.g. 0.2).
* ``--haario`` -- Haario adaptive covariance learned from the running chain
  during burn-in (handles the rho/p anisotropy the fixed step cannot). Seeds
  from ``--ensemble-dir`` if given, else from an isotropic ``--cov-warmup``.

Run from the repo root::

    python MCMC/vanilla_mcmc.py
    python MCMC/vanilla_mcmc.py --steps 40000 --step 0.02 --forward euler
    python MCMC/vanilla_mcmc.py --step 0.05 --fast

    # fixed proposal preconditioned by the ensemble covariance (tune cov-scale)
    python MCMC/vanilla_mcmc.py --ensemble-dir training_data/shock_tube/enkf_ensemble_files --cov-scale 0.2
    # Haario adaptive covariance, cold-started
    python MCMC/vanilla_mcmc.py --haario --steps 40000
    # Haario seeded from the ensemble covariance
    python MCMC/vanilla_mcmc.py --haario --ensemble-dir training_data/shock_tube/enkf_ensemble_files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# This script lives one level under the repo root (MCMC/).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from MCMC.mcmc_common import (  # noqa: E402
    FORWARD_MODES, N_FIELD, N_PARAMS, OBS_ERROR, PARAM_NAMES, TRUTH,
    build_observations, load_ensemble, make_forward, make_plots, prior_bounds,
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


def _cov_chol(C, d):
    """Cholesky of a regularized covariance; diagonal fallback if not PD."""
    reg = 1e-10 * np.trace(C) / d + 1e-12
    try:
        return np.linalg.cholesky(C + reg * np.eye(d))
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.clip(np.diag(C), 1e-12, None)))


def _ensemble_cov_u(directory, to_u):
    """Covariance of the HDF5 ensemble's 4 params, mapped into normalized u-space.

    ``load_ensemble`` reads the *.h5 files into an augmented-state matrix; its top
    N_PARAMS rows are the per-member [p_high, p_low, rho_high, rho_low]. Each is
    pushed through ``to_u`` so the covariance lives in the same [0,1]^4 space the
    sampler proposes in.
    """
    M = load_ensemble(directory)[:N_PARAMS, :]                       # (N_PARAMS x Ne)
    U = np.stack([to_u(M[:, j]) for j in range(M.shape[1])], axis=0)  # (Ne x d)
    return np.cov(U, rowvar=False)


# --------------------------------------------------------------------------- #
# Vanilla random-walk Metropolis in normalized [0, 1]^4 space -- fixed proposal.
# --------------------------------------------------------------------------- #
def run_mcmc(args) -> dict:
    rng = np.random.default_rng(args.seed)

    # Operating point the observations are synthesized from (--truth overrides).
    truth = TRUTH if args.truth is None else np.asarray(args.truth, dtype=float)

    H, d_obs, cell_idx, exact_rho = build_observations(args.obs_every, args.margin,
                                                       truth=truth)
    g = make_forward(args.forward, cell_idx, in_process=args.fast)
    lo, hi = prior_bounds(frac=args.prior_frac, contain=truth)
    span = hi - lo

    # Map physical params <-> normalized u in [0, 1]^4.
    to_u = lambda m: (m - lo) / span
    to_m = lambda u: lo + u * span
    logp = lambda m: log_posterior(m, g, d_obs, args.obs_error, lo, hi)

    # --- proposal setup: isotropic / ensemble-preconditioned / Haario --------
    d = N_PARAMS
    use_ensemble = args.ensemble_dir is not None
    chol = np.eye(d) * args.step                 # default: isotropic scale
    if use_ensemble:                             # seed cov from the HDF5 ensemble
        chol = _cov_chol(_ensemble_cov_u(args.ensemble_dir, to_u), d)
    if args.haario:
        mode = "Haario adaptive cov" + (" (ensemble-seeded)" if use_ensemble else "")
    elif use_ensemble:
        mode = "fixed ensemble-preconditioned cov"
    else:
        mode = f"fixed isotropic step={args.step}"

    print(f"[setup] vanilla MCMC steps={args.steps} burn={args.burn} thin={args.thin} "
          f"forward={args.forward} m={d_obs.size} obs_error={args.obs_error}")
    print(f"[setup] proposal: {mode}")
    print(f"[truth] " + "  ".join(f"{n}={truth[k]:.4e}"
                                   for k, n in enumerate(PARAM_NAMES)))
    print("[prior] uniform bounds (frac={:.2f}):".format(args.prior_frac))
    for k, n in enumerate(PARAM_NAMES):
        print(f"    {n:9s} [{lo[k]:.4e}, {hi[k]:.4e}]")

    # Start at the center of the prior box (u = 0.5 in every normalized dim).
    u = np.full(N_PARAMS, 0.5)
    cur_lp = logp(to_m(u))

    n_total = args.steps
    chain = np.empty((n_total, N_PARAMS))   # stored in PHYSICAL units
    lp_chain = np.empty(n_total)
    n_acc = 0

    # Running moments of u (Welford) -> Haario adaptive covariance.
    mu = u.copy()
    M2 = np.zeros((d, d))
    n_mom = 1

    for i in range(n_total):
        print(f"Step {i}/{n_total}", end="\r")
        z = rng.standard_normal(d)
        # Isotropic step only for plain vanilla, or while Haario warms up its cov;
        # otherwise propose along chol(C) (ensemble seed or learned covariance).
        warming = args.haario and not use_ensemble and i < args.cov_warmup
        if warming or (not args.haario and not use_ensemble):
            prop = u + args.step * z
        else:
            prop = u + args.cov_scale * (chol @ z)
        prop_lp = logp(to_m(prop))
        if np.log(rng.random()) < prop_lp - cur_lp:
            u, cur_lp = prop, prop_lp
            n_acc += 1
        chain[i] = to_m(u)
        lp_chain[i] = cur_lp

        # Update running moments; refresh the Haario covariance during burn-in.
        n_mom += 1
        delta = u - mu
        mu += delta / n_mom
        M2 += np.outer(delta, u - mu)
        if args.haario and i < args.burn and (i + 1) % args.adapt_every == 0 \
                and n_mom > d + 1:
            chol = _cov_chol(M2 / (n_mom - 1), d)

    # Posterior = post-burn-in, thinned.
    post = chain[args.burn::args.thin]
    acc_rate = n_acc / n_total

    mean = post.mean(axis=0)
    std = post.std(axis=0)
    q = np.percentile(post, [2.5, 50, 97.5], axis=0)
    map_idx = int(np.argmax(lp_chain))
    map_params = chain[map_idx]

    print(f"[run] acceptance={acc_rate:.3f}  posterior samples={post.shape[0]}")
    print("[posterior] mean +/- std   [2.5%, 97.5%]   (truth):")
    for k, n in enumerate(PARAM_NAMES):
        print(f"    {n:9s} {mean[k]:.4e} +/- {std[k]:.2e}   "
              f"[{q[0, k]:.4e}, {q[2, k]:.4e}]   (truth {truth[k]:.4e})")

    # Reconstruct full density field from posterior draws -> credible band.
    g_full = make_forward(args.forward, in_process=args.fast)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forward", choices=FORWARD_MODES, default="exact",
                    help="forward model g(m): 'exact' analytic Sod (inverse crime, fast), "
                         "'euler' numerical solver (honest, slower), or 'surrogate' "
                         "trained GPR")
    ap.add_argument("--truth", type=float, nargs=4, default=None,
                    metavar=("P_HIGH", "P_LOW", "RHO_HIGH", "RHO_LOW"),
                    help="operating point the observations are synthesized from "
                         "(default: TRUTH = 1e5 1e4 1.0 0.125)")
    ap.add_argument("--steps", type=int, default=20000, help="total MCMC steps")
    ap.add_argument("--burn", type=int, default=4000, help="burn-in steps to discard")
    ap.add_argument("--thin", type=int, default=5, help="keep every k-th post-burn sample")
    ap.add_argument("--step", type=float, default=0.03,
                    help="FIXED isotropic proposal std in normalized [0,1] space "
                         "(the single knob -- nothing adapts it)")
    # --- optional covariance proposals (no longer strictly 'vanilla') --------
    ap.add_argument("--haario", action="store_true",
                    help="use a Haario adaptive-covariance proposal: learn the "
                         "proposal covariance from the running chain during burn-in "
                         "(handles the rho/p anisotropy the fixed step cannot)")
    ap.add_argument("--ensemble-dir", dest="ensemble_dir", default=None,
                    help="directory of HDF5 ensemble files; seed the proposal "
                         "covariance from the ensemble's parameter covariance. Used "
                         "as a fixed preconditioner on its own, or as the Haario seed")
    ap.add_argument("--cov-scale", dest="cov_scale", type=float,
                    default=2.38 / np.sqrt(N_PARAMS),
                    help="scalar multiplier on chol(C) for covariance proposals "
                         "(2.38/sqrt(d) is the Roberts-Rosenthal optimum)")
    ap.add_argument("--cov-warmup", dest="cov_warmup", type=int, default=300,
                    help="Haario WITHOUT an ensemble seed: take isotropic --step "
                         "steps for this many iters to build the initial covariance")
    ap.add_argument("--adapt-every", dest="adapt_every", type=int, default=100,
                    help="Haario: refresh the proposal covariance every k burn-in steps")
    ap.add_argument("--prior-frac", dest="prior_frac", type=float, default=0.25,
                    help="uniform prior half-width as a fraction of each parameter's "
                         "center, i.e. bounds = center * (1 -/+ frac)")
    ap.add_argument("--n-field-draws", dest="n_field_draws", type=int, default=300,
                    help="posterior draws used to build the reconstructed-field band")
    # Observation methodology -- identical flags to the EnKF drivers.
    ap.add_argument("--obs-every", type=int, default=15,
                    help="baseline: observe every k-th density cell in the flat regions")
    ap.add_argument("--margin", type=int, default=4,
                    help="exclude cells within this many of each discontinuity")
    ap.add_argument("--obs-error", type=float, default=OBS_ERROR,
                    help="observation noise std (likelihood sigma)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    ap.add_argument("--fast", action="store_true",
                    help="testing shortcut: run the exact/euler forward models "
                         "in-process instead of shelling out per sample (much "
                         "faster; the subprocess default mirrors the eventual CFD "
                         "executable)")
    args = ap.parse_args()

    res = run_mcmc(args)
    if not args.no_plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        make_plots(res, outdir, label="vanilla")


if __name__ == "__main__":
    main()
