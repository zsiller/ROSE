#!/usr/bin/env python
"""Write Dakota calibration data and observation cell mask for a Sod param set.

Given the 4 ICs ``[p_high, p_low, rho_high, rho_low]``, build the interior
observation mask (same logic as ``gen_calibration_data.py``) and write:

  * ``sod_obs.dat``   — one freeform row: densities then per-cell variances
  * ``sod_cells.npy`` — integer cell indices loaded by ``sod_driver.py``

Run from anywhere::

    python dakota_mcmc/write_observations.py 0.9e5 0.9e4 0.9 0.1
    python dakota_mcmc/write_observations.py 0.9e5 0.9e4 0.9 0.1 --out-dir /tmp/exp1
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

from common import OBS_ERROR, PARAM_NAMES, build_observations  # noqa: E402

OBS_EVERY = 15
MARGIN = 4
DEFAULT_OBS_OUT = HERE / "sod_obs.dat"
DEFAULT_CELLS_OUT = HERE / "sod_cells.npy"


def write_observations(params: np.ndarray, *,
                       obs_out: Path = DEFAULT_OBS_OUT,
                       cells_out: Path = DEFAULT_CELLS_OUT,
                       obs_error: float = OBS_ERROR,
                       obs_every: int = OBS_EVERY,
                       margin: int = MARGIN,
                       seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build the obs mask at ``params`` and write Dakota calibration files."""
    params = np.asarray(params, dtype=float)
    if params.size != 4:
        raise ValueError(f"expected 4 Sod params, got {params.size}")

    _, obs, cell_idx, _ = build_observations(
        obs_every, margin, truth=params, obs_error=obs_error, seed=seed)
    variances = np.full(obs.size, obs_error ** 2)

    obs_out = Path(obs_out)
    cells_out = Path(cells_out)
    obs_out.parent.mkdir(parents=True, exist_ok=True)
    cells_out.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(obs_out, np.concatenate([obs, variances])[None, :],
               fmt="%.12e", delimiter="  ")
    np.save(cells_out, cell_idx)
    return obs, cell_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", type=float, nargs=4, metavar=tuple(PARAM_NAMES),
                    help="operating point used to place the observation mask")
    ap.add_argument("--out-dir", type=Path, default=HERE,
                    help="directory for sod_obs.dat and sod_cells.npy")
    ap.add_argument("--obs-error", type=float, default=OBS_ERROR,
                    help="observation noise std (variance = obs_error^2)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for the exact + noise draw (default: nondeterministic)")
    args = ap.parse_args()

    obs_out = args.out_dir / "sod_obs.dat"
    cells_out = args.out_dir / "sod_cells.npy"
    params = np.asarray(args.params, dtype=float)
    obs, cell_idx = write_observations(params, obs_out=obs_out, cells_out=cells_out,
                                       obs_error=args.obs_error, seed=args.seed)

    print("[params] " + "  ".join(f"{n}={params[k]:.4e}"
                                   for k, n in enumerate(PARAM_NAMES)))
    print(f"[obs]    m = {obs.size} interior cells {cell_idx.tolist()}")
    print(f"[obs]    sigma = {args.obs_error}  (variance {args.obs_error ** 2:.3e})")
    print(f"[out]    wrote {obs_out}  ({obs.size} densities + {obs.size} variances)")
    print(f"[out]    wrote {cells_out}  ({cell_idx.size} cell indices)")


if __name__ == "__main__":
    main()
