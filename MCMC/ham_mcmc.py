"""Hamiltonian Monte Carlo for the 1D shock tube -- infer the 4 boundary
parameters from final-time observations.

Gradient-based counterpart to ``mcmc_infer.py`` (random-walk Metropolis) on the
exact same inverse problem: sample the posterior over

    m = [p_high, p_low, rho_high, rho_low]

given density observations at ``T_FINAL``. HMC augments the 4 params with a
momentum, then proposes by simulating Hamiltonian dynamics with a leapfrog
integrator -- so a single accepted proposal moves a long, near-energy-conserving
distance across the (strongly anisotropic) posterior instead of taking one small
random-walk step. This typically gives far higher effective sample size per
forward solve than RW-Metropolis, at the cost of needing the log-posterior
GRADIENT.

Gradient
--------
The forward models (exact Sod / Euler solver / GPR surrogate) are not
autodifferentiable here, so the gradient of the potential
``U(m) = -log pi(m | y)`` is taken by finite differences in the normalized
[0,1]^4 space:

    central  (``--fd-mode central``): 2*d forward solves / gradient  (accurate)
    forward  (``--fd-mode forward`` ): d   forward solves / gradient  (cheaper)

Because the observations deliberately sit in the flat regions (fronts excluded,
``--margin``), the misfit is smooth in the params and these differences are
well-behaved. Each leapfrog trajectory costs ``L`` gradients, so HMC does a LOT
of forward solves -- fine for ``--forward exact`` (fast), heavy for
``--forward euler``; start with ``exact`` and a modest ``--n-samples``.

Sampler details
---------------
* Normalized parameter space u in [0,1]^4 (the four scales p~1e5 vs rho~0.1 are
  made comparable, exactly as in ``mcmc_infer.py``).
* Uniform prior over the padded ensemble box -> inside the box the prior is
  constant (zero gradient); the hard box constraint is enforced by REFLECTING the
  leapfrog trajectory at the unit-cube walls (position mirrored, momentum
  negated), which keeps every proposal feasible.
* Diagonal mass matrix M; optionally adapted from the warmup-chain variance
  (``--adapt-mass``) so momentum is scaled to the posterior's per-axis spread.
* Leapfrog step size eps tuned by Nesterov dual averaging toward
  ``--target-accept`` (0.8) during burn-in, then frozen so the retained chain has
  a fixed transition kernel. ``L`` leapfrog steps per trajectory, optionally
  jittered (``--jitter``) to avoid resonant path lengths.

Run from the repo root::

    python inference/mcmc/hmc_infer.py
    python inference/mcmc/hmc_infer.py --n-samples 600 --forward euler --fd-mode forward
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
    ENSEMBLE_DIR, FORWARD_MODES, N_FIELD, N_PARAMS, OBS_ERROR, PARAM_NAMES,
    TRUTH,
    build_observations, load_ensemble, make_forward, make_plots, prior_bounds,
)


# --------------------------------------------------------------------------- #
# Potential U(u) = -log posterior, and its finite-difference gradient.
# Inside the uniform-prior box the prior is constant, so U is just the scaled
# data misfit; the box itself is enforced by reflection in the leapfrog, so the
# samplers below only ever call these on feasible u.
# --------------------------------------------------------------------------- #
def make_potential(g, d_obs, sigma, lo, hi):
    """Return (potential, to_m) where potential(u) = 0.5*||g(m)-d||^2/sigma^2."""
    span = hi - lo
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    def to_m(u):
        return lo + u * span

    def potential(u):
        resid = g(to_m(u)) - d_obs
        return float(np.dot(resid, resid) * inv2s2)

    return potential, to_m


def grad_potential(potential, u, U0, fd_step, mode):
    """Finite-difference gradient of `potential` at u (in normalized space).

    `central`: 2*d evals, O(h^2) accurate.  `forward`: d evals reusing U0=U(u).
    """
    d = u.size
    grad = np.empty(d)
    if mode == "central":
        for k in range(d):
            up = u.copy(); up[k] += fd_step
            um = u.copy(); um[k] -= fd_step
            grad[k] = (potential(up) - potential(um)) / (2.0 * fd_step)
    else:  # forward
        for k in range(d):
            up = u.copy(); up[k] += fd_step
            grad[k] = (potential(up) - U0) / fd_step
    return grad


# --------------------------------------------------------------------------- #
# Reflective leapfrog: keep u in the unit cube [0,1]^d by mirroring at the walls.
# --------------------------------------------------------------------------- #
def _reflect(u, p):
    """Reflect position u into [0,1]^d, negating the matching momentum components.

    Loops in case a single step overshoots a wall by more than one cube width
    (rare for sane step sizes, but keeps the map well-defined).
    """
    while True:
        below = u < 0.0
        above = u > 1.0
        if not (below.any() or above.any()):
            return u, p
        u = np.where(below, -u, u)
        u = np.where(above, 2.0 - u, u)
        p = np.where(below | above, -p, p)


def leapfrog(u, p, potential, eps, L, minv, fd_step, fd_mode):
    """L reflective leapfrog steps. Returns (u, p, grad_evals)."""
    g0 = grad_potential(potential, u, potential(u), fd_step, fd_mode)
    p = p - 0.5 * eps * g0
    evals = 1
    for i in range(L):
        u = u + eps * (minv * p)
        u, p = _reflect(u, p)
        gi = grad_potential(potential, u, potential(u), fd_step, fd_mode)
        evals += 1
        if i != L - 1:
            p = p - eps * gi
        else:
            p = p - 0.5 * eps * gi
    return u, -p, evals  # negate momentum -> reversible proposal


# --------------------------------------------------------------------------- #
# Effective sample size (for comparing mixing against RW-Metropolis).
# --------------------------------------------------------------------------- #
def ess_1d(x):
    """Effective sample size of a 1-D chain via the initial-positive-sequence."""
    n = x.size
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0.0:
        return float(n)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (var * n)
    s, t = 1.0, 1
    while t < n - 1:
        pair = acf[t] + acf[t + 1] if t + 1 < n else acf[t]
        if pair <= 0:
            break
        s += 2.0 * acf[t]
        t += 1
    return float(n / s)


# --------------------------------------------------------------------------- #
# HMC sampler in normalized [0,1]^4 space.
# --------------------------------------------------------------------------- #
def run_hmc(args) -> dict:
    rng = np.random.default_rng(args.seed)
    directory = Path(args.dir)

    truth = TRUTH if args.truth is None else np.asarray(args.truth, dtype=float)

    H, d_obs, cell_idx, exact_rho = build_observations(args.obs_every, args.margin,
                                                       truth=truth)
    g = make_forward(args.forward, cell_idx)
    lo, hi = prior_bounds(directory, args.prior_pad, contain=truth)
    span = hi - lo
    to_u = lambda m: (m - lo) / span

    potential, to_m = make_potential(g, d_obs, args.obs_error, lo, hi)

    d = N_PARAMS
    n_total = args.n_samples + args.burn

    print(f"[setup] HMC samples={args.n_samples} burn={args.burn} L={args.L} "
          f"forward={args.forward} fd={args.fd_mode} m={d_obs.size} "
          f"obs_error={args.obs_error}")
    print(f"[truth] " + "  ".join(f"{n}={truth[k]:.4e}"
                                  for k, n in enumerate(PARAM_NAMES)))
    print("[prior] uniform bounds (pad={:.2f}):".format(args.prior_pad))
    for k, n in enumerate(PARAM_NAMES):
        print(f"    {n:9s} [{lo[k]:.4e}, {hi[k]:.4e}]")

    # Start at the ensemble mean, mapped into the unit cube.
    u = np.clip(to_u(load_ensemble(directory)[:N_PARAMS, :].mean(axis=1)),
                1e-6, 1.0 - 1e-6)

    # Diagonal mass matrix (M); minv = 1/M used in kinetic energy & dynamics.
    mass = np.ones(d)
    minv = 1.0 / mass

    # Nesterov dual averaging for the step size eps (Hoffman & Gelman 2014).
    eps = args.step
    mu_da = np.log(10.0 * eps)
    log_eps_bar, H_bar = 0.0, 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75

    chain = np.empty((n_total, d))      # physical units
    lp_chain = np.empty(n_total)
    u_chain = np.empty((n_total, d))    # normalized units (for mass adaptation/ESS)
    n_acc = 0
    total_evals = 0

    cur_U = potential(u)
    for i in range(n_total):
        print(f"Sample {i+1}/{n_total}", end="\r")
        # Sample momentum p ~ N(0, M); kinetic K = 0.5 p^T M^-1 p.
        p0 = rng.standard_normal(d) * np.sqrt(mass)
        K0 = 0.5 * np.dot(p0, minv * p0)

        Li = args.L if not args.jitter else int(rng.integers(
            max(1, args.L - args.jitter), args.L + args.jitter + 1))
        u_new, p_new, evals = leapfrog(u, p0, potential, eps, Li,
                                       minv, args.fd_step, args.fd_mode)
        total_evals += evals

        U_new = potential(u_new)
        K_new = 0.5 * np.dot(p_new, minv * p_new)
        # Metropolis on the Hamiltonian H = U + K.
        log_accept = (cur_U + K0) - (U_new + K_new)
        accept_prob = float(min(1.0, np.exp(min(0.0, log_accept))))
        if np.log(rng.random()) < log_accept:
            u, cur_U = u_new, U_new
            n_acc += 1

        chain[i] = to_m(u)
        u_chain[i] = u
        lp_chain[i] = -cur_U

        # --- dual-averaging step-size adaptation (burn-in only) -------------
        if args.adapt and i < args.burn:
            m1 = i + 1.0
            H_bar = (1.0 - 1.0 / (m1 + t0)) * H_bar \
                + (1.0 / (m1 + t0)) * (args.target_accept - accept_prob)
            log_eps = mu_da - (np.sqrt(m1) / gamma) * H_bar
            eta = m1 ** (-kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            eps = float(np.exp(log_eps))
        elif args.adapt and i == args.burn:
            eps = float(np.exp(log_eps_bar))  # freeze at the averaged value

        # --- diagonal mass adaptation: refresh once, midway through burn-in -
        if (args.adapt_mass and i == args.burn // 2 and i > d + 1):
            v = u_chain[:i + 1].var(axis=0)
            v = np.clip(v, 1e-8, None)
            mass = 1.0 / v          # M = diag(1/Var(u)) -> whitens the momentum
            minv = 1.0 / mass

    post = chain[args.burn:]
    u_post = u_chain[args.burn:]
    acc_rate = n_acc / n_total

    mean = post.mean(axis=0)
    std = post.std(axis=0)
    q = np.percentile(post, [2.5, 50, 97.5], axis=0)
    map_idx = int(np.argmax(lp_chain))
    map_params = chain[map_idx]
    ess = np.array([ess_1d(u_post[:, k]) for k in range(d)])

    print(f"[run] acceptance={acc_rate:.3f}  final eps={eps:.4e}  "
          f"grad-solves={total_evals}  posterior samples={post.shape[0]}")
    print("[posterior] mean +/- std   [2.5%, 97.5%]   ESS   (truth):")
    for k, n in enumerate(PARAM_NAMES):
        print(f"    {n:9s} {mean[k]:.4e} +/- {std[k]:.2e}   "
              f"[{q[0, k]:.4e}, {q[2, k]:.4e}]   ESS={ess[k]:6.0f}   "
              f"(truth {truth[k]:.4e})")

    # Posterior-predictive density field (push draws through the full forward).
    g_full = make_forward(args.forward)
    draw_idx = rng.choice(post.shape[0],
                          size=min(args.n_field_draws, post.shape[0]),
                          replace=False)
    fields = np.array([g_full(post[j]) for j in draw_idx])
    field_mean = fields.mean(axis=0)
    field_lo, field_hi = np.percentile(fields, [2.5, 97.5], axis=0)
    post_rmse = float(np.sqrt(np.mean((field_mean - exact_rho) ** 2)))
    print(f"[result] posterior-mean density RMSE vs exact = {post_rmse:.4e}")

    return {
        "chain": chain, "lp_chain": lp_chain, "post": post, "truth": truth,
        "mean": mean, "std": std, "quantiles": q, "ess": ess,
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
    ap.add_argument("--dir", default=str(ENSEMBLE_DIR),
                    help="directory of the 20 ensemble HDF5 files (for prior bounds)")
    ap.add_argument("--forward", choices=FORWARD_MODES, default="exact",
                    help="forward model g(m): 'exact' analytic Sod (fast), 'euler' "
                         "numerical solver (honest, much slower under HMC's many "
                         "grads), or 'surrogate' trained GPR")
    ap.add_argument("--truth", type=float, nargs=4, default=None,
                    metavar=("P_HIGH", "P_LOW", "RHO_HIGH", "RHO_LOW"),
                    help="operating point the observations are synthesized from "
                         "(default: common.TRUTH). Prior box auto-expands to contain it.")
    # HMC controls.
    ap.add_argument("--n-samples", dest="n_samples", type=int, default=1000,
                    help="retained HMC samples (post burn-in)")
    ap.add_argument("--burn", type=int, default=400, help="warmup samples to discard")
    ap.add_argument("--L", type=int, default=20, help="leapfrog steps per trajectory")
    ap.add_argument("--jitter", type=int, default=5,
                    help="randomize L within +/- this many steps (0 = fixed L)")
    ap.add_argument("--step", type=float, default=0.02,
                    help="initial leapfrog step size eps in normalized space")
    ap.add_argument("--target-accept", dest="target_accept", type=float, default=0.8,
                    help="dual-averaging target acceptance probability")
    ap.add_argument("--no-adapt", dest="adapt", action="store_false",
                    help="disable dual-averaging step-size adaptation")
    ap.add_argument("--adapt-mass", dest="adapt_mass", action="store_true",
                    help="adapt a diagonal mass matrix from warmup-chain variance")
    # Gradient.
    ap.add_argument("--fd-mode", dest="fd_mode", choices=["central", "forward"],
                    default="central",
                    help="finite-difference gradient: 'central' (2d solves, accurate) "
                         "or 'forward' (d solves, cheaper -- good for euler)")
    ap.add_argument("--fd-step", dest="fd_step", type=float, default=1e-4,
                    help="finite-difference step in normalized [0,1] space")
    ap.add_argument("--prior-pad", dest="prior_pad", type=float, default=0.25,
                    help="pad the ensemble min/max prior box by this fraction of its half-width")
    ap.add_argument("--n-field-draws", dest="n_field_draws", type=int, default=300,
                    help="posterior draws used to build the reconstructed-field band")
    # Observation methodology -- identical flags to mcmc_infer / enkf_driver.
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

    res = run_hmc(args)
    if not args.no_plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # Reuse mcmc_infer's plotter; tag outputs as HMC so they don't clobber RWM.
        make_plots(res, outdir)
        for stem in ("mcmc_marginals", "mcmc_field"):
            src = outdir / f"{stem}.png"
            if src.exists():
                dst = outdir / f"{stem.replace('mcmc', 'hmc')}.png"
                src.replace(dst)
                print(f"[plot] renamed -> {dst}")


if __name__ == "__main__":
    main()
