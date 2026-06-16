#!/usr/bin/env python
"""Write a starting proposal covariance for QUESO from an HDF5 ensemble.

QUESO's random-walk Metropolis needs a proposal covariance that spans the 4-5
orders of magnitude between the pressure params (~1e5) and the densities (~1).
A good *prior* starting point -- available before any chain exists -- is the
empirical covariance of the 4 Sod ICs across a precomputed ENSEMBLE of forward
runs stored as HDF5 files (each file carries the 4 ICs as attributes).

This reuses ``common.load_ensemble``, which reads a directory of
``*.h5`` ensemble files into an ``(N_STATE x Ne)`` augmented-state matrix whose
top ``N_PARAMS`` rows are ``[p_high, p_low, rho_high, rho_low]``. We take those
rows, compute their covariance across the ensemble members, scale, and write
``prop_cov.dat`` as a freeform 4x4 matrix for::

    proposal_covariance
      file 'prop_cov.dat'
        matrix

    python dakota_mcmc/gen_proposal_cov.py
    python dakota_mcmc/gen_proposal_cov.py --ensemble-dir <dir> --scale 1.42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ENSEMBLE_DIR, N_PARAMS, PARAM_NAMES, load_ensemble,
)

HERE = Path(__file__).resolve().parent
DEFAULT_COV_OUT = HERE / "prop_cov.dat"


def write_proposal_cov(*,
                       ensemble_dir: Path | str = ENSEMBLE_DIR,
                       scale: float = 1.0,
                       out: Path | str = DEFAULT_COV_OUT) -> np.ndarray:
    """Empirical 4x4 proposal covariance from the HDF5 ensemble -> ``prop_cov.dat``."""
    out = Path(out)
    X = load_ensemble(ensemble_dir)
    params = X[:N_PARAMS, :]
    ne = params.shape[1]
    if ne < N_PARAMS + 1:
        raise ValueError(f"need >= {N_PARAMS + 1} members for a full-rank 4x4 "
                         f"covariance; got {ne}")

    cov = np.cov(params) * scale
    np.savetxt(out, cov, fmt="%.12e")
    return cov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-dir", dest="ensemble_dir", default=str(ENSEMBLE_DIR),
                    help="directory of *.h5 ensemble files (4 ICs as attrs per file)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the covariance by this (use 2.38^2/d = 1.42 for "
                         "the optimal random-walk scaling; >1 widens the proposal "
                         "to improve mixing)")
    ap.add_argument("--out", default=str(DEFAULT_COV_OUT))
    args = ap.parse_args()

    X = load_ensemble(args.ensemble_dir)
    params = X[:N_PARAMS, :]
    ne = params.shape[1]
    cov = write_proposal_cov(ensemble_dir=args.ensemble_dir,
                             scale=args.scale, out=args.out)

    print(f"[ensemble] {args.ensemble_dir}: {ne} members")
    print(f"[scale] x{args.scale}")
    print("[mean]  " + "  ".join(f"{n}={params[k].mean():.4e}"
                                 for k, n in enumerate(PARAM_NAMES)))
    print("[cov] proposal covariance (4x4):")
    for i, name in enumerate(PARAM_NAMES):
        print(f"    {name:9s} " + "  ".join(f"{cov[i, j]: .4e}" for j in range(N_PARAMS)))
    print("[std]  sqrt(diag) = " + "  ".join(
        f"{n}={np.sqrt(cov[k, k]):.3e}" for k, n in enumerate(PARAM_NAMES)))
    print(f"[out]  wrote {args.out}  (freeform {N_PARAMS}x{N_PARAMS} matrix)")


if __name__ == "__main__":
    main()
