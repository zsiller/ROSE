#!/usr/bin/env python
"""Plot posterior marginals from a Dakota chain (``sod_chain.dat``).

Thin wrapper around ``common.make_plots`` (marginals + field/residuals).
Prefer ``run_mcmc.py --plot`` for the full workflow.

    python dakota_mcmc/plot_marginals.py
    python dakota_mcmc/plot_marginals.py --chain sod_chain.dat --burn 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    N_FIELD, PARAM_NAMES, T_FINAL, TRUTH, exact_density_on_cells,
    make_forward, make_plots,
)


def load_chain(path: Path) -> np.ndarray:
    with open(path) as fh:
        header = fh.readline().lstrip("%").split()
    cols = [header.index(name) for name in PARAM_NAMES]
    return np.atleast_2d(np.loadtxt(path, skiprows=1, usecols=cols))


def load_obs_densities(path: Path) -> np.ndarray:
    row = np.atleast_1d(np.loadtxt(path))
    return row[: row.size // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default=str(HERE / "sod_chain.dat"))
    ap.add_argument("--obs", default=str(HERE / "sod_obs.dat"))
    ap.add_argument("--cells", default=str(HERE / "sod_cells.npy"))
    ap.add_argument("--burn", type=int, default=0,
                    help="number of chain samples to discard as burn-in")
    ap.add_argument("--truth", type=float, nargs=4, default=None,
                    metavar=tuple(PARAM_NAMES))
    ap.add_argument("--forward", choices=("exact", "euler", "surrogate"),
                    default="euler")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    chain = load_chain(Path(args.chain))
    burn = min(args.burn, chain.shape[0])
    post = chain[burn:]
    truth = TRUTH if args.truth is None else np.asarray(args.truth, dtype=float)

    mean = post.mean(axis=0)
    std = post.std(axis=0)
    quantiles = np.percentile(post, [2.5, 50, 97.5], axis=0)

    cell_idx = np.load(args.cells)
    obs_y = load_obs_densities(Path(args.obs))
    g_full = make_forward(args.forward, in_process=True)
    fields = np.array([g_full(post[j]) for j in range(min(200, post.shape[0]))])
    field_mean = fields.mean(axis=0)
    field_lo, field_hi = np.percentile(fields, [2.5, 97.5], axis=0)
    exact_rho = exact_density_on_cells(truth, T_FINAL, N_FIELD)
    post_rmse = float(np.sqrt(np.mean((field_mean - exact_rho) ** 2)))

    print(f"[chain] {chain.shape[0]} samples, burn {burn} -> {post.shape[0]} kept")
    for k, name in enumerate(PARAM_NAMES):
        print(f"    {name:9s} {mean[k]:.4e} +/- {std[k]:.2e}   "
              f"[{quantiles[0, k]:.4e}, {quantiles[2, k]:.4e}]   "
              f"(truth {truth[k]:.4e})")

    make_plots({
        "chain": chain, "post": post, "truth": truth,
        "mean": mean, "std": std, "quantiles": quantiles,
        "acc_rate": 0.0,
        "x_cells": np.arange(N_FIELD) / N_FIELD,
        "exact": exact_rho, "field_mean": field_mean,
        "field_lo": field_lo, "field_hi": field_hi,
        "obs_x": cell_idx / N_FIELD, "obs_y": obs_y,
        "post_rmse": post_rmse, "burn": burn, "forward": args.forward,
    }, args.out_dir, label="dakota")


if __name__ == "__main__":
    main()
