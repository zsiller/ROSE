#!/usr/bin/env python
"""Surrogate-vs-observations validation using the ``scores`` library's CRPS.

Same pipeline as ``validation.py`` (push a Dakota MCMC chain through the GPR
surrogate at the external obs cells), but the per-cell CRPS is computed with
``scores.probability.crps_for_ensemble`` instead of the hand-rolled estimator.
The data loaders, surrogate evaluation, constants, and the figure all come from
``validation.py`` so this script only swaps the scoring backend.

``scores`` is xarray-native and offers two ensemble estimators:
  * ``method="ecdf"``  -> exact CRPS of the empirical CDF (K = M^2)
  * ``method="fair"``  -> unbiased E|X-y| - 1/2 E|X-X'| (K = M(M-1))
``fair`` matches the proper-score interpretation used in ``validation.py`` and
is the default here. NOTE: ``crps_for_ensemble`` scores the raw ensemble against
a point observation, so it does NOT fold in the per-cell observation variance
the way ``validation.py``'s Gaussian-kernel path does.

Run from the repo root::

    python validation/validation_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scores.probability import crps_for_ensemble  # noqa: E402

from helpers.inverse_common import exact_density_on_cells  # noqa: E402
import validation as val  # noqa: E402  (loaders + constants + surrogate eval)

CRPS_METHOD = "fair"          # "fair" (unbiased) or "ecdf" (exact empirical CDF)
OUT = HERE / "surrogate_obs_dist_scores.png"
CRPS_DAT = HERE / "surrogate_crps_scores.dat"
NO_PLOT = False


def crps_scores(pred: np.ndarray, obs: np.ndarray, cell_idx: np.ndarray,
                method: str = CRPS_METHOD) -> np.ndarray:
    """Per-cell CRPS via scores.crps_for_ensemble. pred: (n_samples, m)."""
    fcst = xr.DataArray(pred, dims=["sample", "cell"], coords={"cell": cell_idx})
    obs_da = xr.DataArray(obs, dims=["cell"], coords={"cell": cell_idx})
    crps_da = crps_for_ensemble(fcst, obs_da, ensemble_member_dim="sample",
                                method=method, preserve_dims=["cell"])
    return np.asarray(crps_da.values, dtype=float)


if __name__ == "__main__":
    obs, _variances = val.load_calibration_obs(val.OBS)
    cell_idx = val.load_cell_indices(val.CELLS, obs.size)
    obs_x = cell_idx / val.N_FIELD

    chain = val.load_chain_params(val.CHAIN)
    fields = val.surrogate_fields(chain, val.T_FINAL, val.SURROGATE)
    pred = fields[:, cell_idx]

    pmean = pred.mean(axis=0)
    plo, pmed, phi = np.percentile(pred, [2.5, 50, 97.5], axis=0)
    covered = (obs >= plo) & (obs <= phi)
    flo, fmed, fhi = np.percentile(fields, [2.5, 50, 97.5], axis=0)

    crps = crps_scores(pred, obs, cell_idx, method=CRPS_METHOD)

    print(f"[obs]    {val.OBS}: m = {obs.size} cells {cell_idx.tolist()}")
    print(f"[chain]  {val.CHAIN}: {chain.shape[0]} draws "
          f"(burn {val.BURN:.0%}, cap {val.MAX_SAMPLES})")
    print(f"[surrogate] {val.SURROGATE}")
    print(f"[result] surrogate-pred mean RMSE vs observed = "
          f"{np.sqrt(np.mean((pmean - obs) ** 2)):.4e}")
    print(f"[result] observed value inside 95% predictive band at "
          f"{covered.sum()}/{covered.size} obs points")
    print(f"[crps] backend=scores ({CRPS_METHOD}); mean CRPS = {crps.mean():.4e} "
          f"(lower is better; density units)")
    print(f"[crps] per-cell min/max = {crps.min():.4e} / {crps.max():.4e}")

    np.savetxt(
        CRPS_DAT,
        np.column_stack([cell_idx, obs_x, obs, pmed, crps]),
        fmt=["%d", "%.6f", "%.8e", "%.8e", "%.8e"],
        header=f"per-cell CRPS via scores ({CRPS_METHOD})  "
               f"(mean {crps.mean():.6e}, {chain.shape[0]} draws, "
               f"t={val.T_FINAL:.4e})\ncell_idx  obs_x  obs  surrogate_median  crps",
    )
    print(f"[crps] wrote per-cell scores to {CRPS_DAT}")

    if NO_PLOT:
        raise SystemExit(0)

    from plots import plot_crps

    exact_rho = exact_density_on_cells(np.median(chain, axis=0), val.T_FINAL,
                                       val.N_FIELD)

    plot_crps(OUT, obs_x=obs_x, cell_idx=cell_idx, crps=crps,
              exact_rho=exact_rho, pmed=pmed, fmed=fmed, flo=flo, fhi=fhi,
              obs=obs, n_field=val.N_FIELD,
              t=val.T_FINAL, chain_name=val.CHAIN.name)
