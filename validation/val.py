#!/usr/bin/env python
"""Push a Dakota MCMC chain through the GPR surrogate and show the spread of
surrogate predictions at each observation point.

Given a Dakota ``export_chain_points_file`` (e.g. ``sod_chain.dat``), each row
holds a posterior draw of the 4 Sod ICs ``[p_high, p_low, rho_high, rho_low]``.
We evaluate the trained surrogate at every draw -> a 256-cell density field per
sample -> and keep only the cells that were actually OBSERVED during the
calibration. Stacking those over the whole chain gives, at each observation
point, the surrogate's posterior-predictive distribution (parameter uncertainty
propagated through the emulator).

It also runs a Probability Integral Transform (PIT) calibration check: the
observed value is transformed through the predictive CDF at each location, and
the resulting values should be uniform on [0, 1] if the pushforward is well
calibrated (U-shape = over-confident, hump = under-confident, slope = biased).

This module owns the compute path; the figures live in ``plots.py``
(``plot_field_violins`` + ``plot_obs_histograms`` + ``plot_pit``).

Run from the repo root::

    python validation/val.py
    python validation/val.py --chain dakota_mcmc/sod_chain_exact.dat
    python validation/val.py --truth 0.9e5 0.9e4 0.9 0.1 --max-samples 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MCMC.mcmc_common import (  # noqa: E402
    N_FIELD, OBS_ERROR, PARAM_NAMES, SURROGATE_PKL, T_FINAL, TRUTH,
    build_observations, load_surrogate,
)


def load_chain_params(path: Path) -> np.ndarray:
    """Read the 4 Sod-param columns from a Dakota exported chain file.

    Header is ``%mcmc_id interface p_high p_low rho_high rho_low <responses>``.
    """
    with open(path) as fh:
        header = fh.readline().lstrip("%").split()
    cols = [header.index(name) for name in PARAM_NAMES]
    return np.atleast_2d(np.loadtxt(path, skiprows=1, usecols=cols))


def surrogate_fields(params: np.ndarray, t: float, pkl: Path | None) -> np.ndarray:
    """Surrogate density field for every param row -> (n_samples, N_FIELD).

    Batched through Surrogate.predict in one shot (POD + GP handle 2-D X).
    """
    sur = load_surrogate(pkl)
    X = np.column_stack([params, np.full(params.shape[0], t)])  # [.., p, t]
    Y, _ = sur.predict(X)
    return np.asarray(Y, dtype=float)[:, :N_FIELD]


def pit_values(pred: np.ndarray, obs: np.ndarray, obs_error: float,
               rng: np.random.Generator) -> np.ndarray:
    """Probability Integral Transform at each location: P(prediction <= obs).

    ``pred`` is (n_samples, m). With ``obs_error > 0`` the predictive of the
    OBSERVABLE is the pushforward + N(0, obs_error^2) (randomised PIT), matching
    the noise the calibration likelihood assumed. If the pushforward is
    well-calibrated the PIT values across locations are uniform on [0, 1].
    """
    samples = pred
    if obs_error > 0.0:
        samples = pred + rng.normal(0.0, obs_error, size=pred.shape)
    return (samples <= obs[None, :]).mean(axis=0)


def uniformity_ks(pit: np.ndarray) -> str:
    """KS statistic of the PIT values against U[0,1], as a printable string."""
    try:
        from scipy import stats
        ks_stat, ks_p = stats.kstest(pit, "uniform")
        return f"KS={ks_stat:.3f}  p={ks_p:.3f}"
    except Exception:
        m = pit.size
        ecdf = np.arange(1, m + 1) / m
        ks_stat = float(np.max(np.abs(ecdf - np.sort(pit))))
        return f"KS={ks_stat:.3f}  p=n/a"


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default=str(ROOT / "dakota_mcmc" / "sod_chain.dat"),
                    help="Dakota exported chain file (param columns p_high..rho_low)")
    ap.add_argument("--surrogate", default=str(SURROGATE_PKL),
                    help="surrogate .pkl (default: the run_200 campaign surrogate)")
    ap.add_argument("--burn", type=float, default=0.2,
                    help="fraction of the chain to discard as burn-in")
    ap.add_argument("--max-samples", dest="max_samples", type=int, default=1500,
                    help="cap on surrogate evaluations (random subsample if exceeded)")
    ap.add_argument("--truth", type=float, nargs=4, default=None,
                    metavar=("P_HIGH", "P_LOW", "RHO_HIGH", "RHO_LOW"),
                    help="operating point the observations were synthesized from "
                         "(sets the observed cells + overlaid data; default TRUTH). "
                         "Match this to gen_calibration_data.py if you changed it.")
    ap.add_argument("--obs-every", type=int, default=15)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--obs-error", dest="obs_error", type=float, default=OBS_ERROR,
                    help="observation-noise std added to the predictive for the PIT "
                         "(0 = raw pushforward PIT)")
    ap.add_argument("--t", type=float, default=T_FINAL, help="snapshot time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(here / "surrogate_obs_dist.png"))
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    args = ap.parse_args()

    truth = TRUTH if args.truth is None else np.asarray(args.truth, dtype=float)

    # Observation geometry + the data, straight from the shared builder.
    _, obs, cell_idx, exact_rho = build_observations(args.obs_every, args.margin,
                                                     truth=truth, t=args.t)
    obs_x = cell_idx / N_FIELD

    # Chain -> param draws (post burn-in, optionally subsampled for speed).
    chain = load_chain_params(Path(args.chain))
    post = chain[int(args.burn * chain.shape[0]):]
    rng = np.random.default_rng(args.seed)
    if post.shape[0] > args.max_samples:
        post = post[rng.choice(post.shape[0], args.max_samples, replace=False)]

    print(f"[chain] {args.chain}: {chain.shape[0]} rows -> {post.shape[0]} "
          f"evaluated (burn {args.burn:.0%}, cap {args.max_samples})")
    print(f"[surrogate] {args.surrogate}")
    print(f"[obs] m = {cell_idx.size} cells at t = {args.t:.4e}; truth "
          + "  ".join(f"{n}={truth[k]:.4e}" for k, n in enumerate(PARAM_NAMES)))

    # Surrogate predictions at the observed cells: (n_samples, m).
    fields = surrogate_fields(post, args.t, Path(args.surrogate))
    pred = fields[:, cell_idx]

    pmean = pred.mean(axis=0)
    plo, pmed, phi = np.percentile(pred, [2.5, 50, 97.5], axis=0)
    covered = (obs >= plo) & (obs <= phi)
    print(f"[result] surrogate-pred mean RMSE vs observed = "
          f"{np.sqrt(np.mean((pmean - obs) ** 2)):.4e}")
    print(f"[result] observed value inside 95% predictive band at "
          f"{covered.sum()}/{covered.size} obs points")

    # PIT calibration of the pushforward against the observations.
    pit = pit_values(pred, obs, args.obs_error, rng)
    ks_str = uniformity_ks(pit)
    print(f"[pit] obs_error added = {args.obs_error};  mean={pit.mean():.3f} "
          f"(ideal 0.5)  std={pit.std():.3f} (ideal {1/np.sqrt(12):.3f})")
    print(f"[pit] uniformity: {ks_str}")

    if args.no_plot:
        return

    from plots import plot_field_violins, plot_obs_histograms, plot_pit

    out = Path(args.out)
    plot_field_violins(out, obs_x=obs_x, pred=pred, obs=obs, pmed=pmed,
                       exact_rho=exact_rho, n_field=N_FIELD,
                       obs_every=args.obs_every, t=args.t, n_draws=post.shape[0])
    plot_obs_histograms(out.with_name(out.stem + "_hist" + out.suffix),
                        cell_idx=cell_idx, obs_x=obs_x, pred=pred, obs=obs,
                        pmed=pmed, plo=plo, phi=phi, t=args.t,
                        n_draws=post.shape[0])
    plot_pit(out.with_name(out.stem + "_pit" + out.suffix), pit=pit,
             ks_str=ks_str, chain_name=Path(args.chain).name)


if __name__ == "__main__":
    main()
