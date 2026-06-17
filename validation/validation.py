#!/usr/bin/env python
"""Push a Dakota MCMC chain through the GPR surrogate at external obs locations.

Reads the calibration data, cell mask, and posterior chain produced by the
Dakota workflow (``write_observations.py`` + ``run_mcmc.py``) instead of
rebuilding the observation mask from a truth vector like ``val.py``.

Run from the repo root::

    python validation/validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.inverse_common import (  # noqa: E402
    exact_density_on_cells, load_surrogate,
)

from helpers.inverse_common import N_FIELD, PARAM_NAMES, T_FINAL  # noqa: E402

SURROGATE_PKL = ROOT / "training_runs" / "shock_tube" / "run_200" / "wf_0" / "surrogate.pkl"

HERE = Path(__file__).resolve().parent
DAKOTA_DIR = ROOT / "dakota_mcmc"
CHAIN = DAKOTA_DIR / "sod_chain.dat"
OBS = DAKOTA_DIR / "sod_obs.dat"
CELLS = DAKOTA_DIR / "sod_cells.npy"
SURROGATE = SURROGATE_PKL
OUT = HERE / "surrogate_obs_dist.png"
CRPS_DAT = HERE / "surrogate_crps.dat"

MAX_SAMPLES = 1500
SEED = 0
NO_PLOT = False


def load_calibration_obs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load densities and per-cell variances from a Dakota freeform obs file."""
    row = np.atleast_1d(np.loadtxt(path))
    if row.size % 2:
        raise ValueError(f"{path}: expected m densities + m variances, got {row.size} cols")
    m = row.size // 2
    return row[:m], row[m:]


def load_cell_indices(path: Path, n_obs: int) -> np.ndarray:
    """Load integer cell indices and check they match the obs file."""
    cell_idx = np.load(path)
    if cell_idx.size != n_obs:
        raise ValueError(f"{path} has {cell_idx.size} indices but obs file has {n_obs}")
    return np.asarray(cell_idx, dtype=int)


def load_chain_params(path: Path, *,
                      max_samples: int = MAX_SAMPLES,
                      seed: int = SEED) -> np.ndarray:
    """Read Sod-param columns from a Dakota exported chain file."""
    with open(path) as fh:
        header = fh.readline().lstrip("%").split()
    cols = [header.index(name) for name in PARAM_NAMES]
    chain = np.atleast_2d(np.loadtxt(path, skiprows=1, usecols=cols))
    
    rng = np.random.default_rng(seed)
    if chain.shape[0] > max_samples:
        chain = chain[rng.choice(chain.shape[0], max_samples, replace=False)]
    return chain


def surrogate_fields(params: np.ndarray, t: float, pkl: Path) -> np.ndarray:
    """Surrogate density field for every param row -> (n_samples, N_FIELD)."""
    sur = load_surrogate(pkl)
    X = np.column_stack([params, np.full(params.shape[0], t)])
    Y, _ = sur.predict(X)
    return np.asarray(Y, dtype=float)[:, :N_FIELD]


def _kernel_term(d: np.ndarray, sig: float) -> np.ndarray:
    """A(d, sig) = E|N(d, sig^2)| = d(2Phi(d/sig) - 1) + 2 sig phi(d/sig).

    The building block of the Gaussian-kernel CRPS. As sig -> 0 it collapses to
    |d|, recovering the bare-ensemble formula.
    """
    if sig <= 0.0:
        return np.abs(d)
    from math import sqrt, pi
    from scipy.special import erf
    z = d / sig
    phi = np.exp(-0.5 * z * z) / sqrt(2.0 * pi)
    Phi = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return d * (2.0 * Phi - 1.0) + 2.0 * sig * phi


def crps_ensemble(pred: np.ndarray, obs: np.ndarray,
                  obs_std: np.ndarray | None = None) -> np.ndarray:
    """Per-obs-point CRPS from an ensemble (proper score, lower is better).

    ``pred`` is (n_samples, m), ``obs`` is (m,). Returns (m,) CRPS in density
    units. For a deterministic forecast CRPS reduces to |pred - obs| (MAE), so
    it subsumes the point error while rewarding a sharp, well-calibrated spread.

    Without ``obs_std`` the bare-ensemble estimator is used (sorted-ensemble
    identity, O(n log n)). With ``obs_std`` (per-cell, shape (m,)) each member is
    broadened into a Gaussian of that width and the *analytic* Gaussian-mixture
    CRPS (Grimit et al. 2006) is evaluated. This folds the known measurement
    variance in deterministically -- the observations are already noisy draws,
    so they are NOT re-perturbed; we only widen the predictive distribution by
    the obs uncertainty before scoring.
    """
    n = pred.shape[0]
    if obs_std is None:
        term1 = np.abs(pred - obs[None, :]).mean(axis=0)      # E|X - y|
        ps = np.sort(pred, axis=0)
        weights = (2 * np.arange(1, n + 1) - n - 1)[:, None]
        return term1 - (weights * ps).sum(axis=0) / (n * n)   # - 0.5 E|X - X'|

    from math import sqrt
    m = pred.shape[1]
    crps = np.empty(m)
    for j in range(m):                                        # per cell: bounds memory
        mu = pred[:, j]
        sig = float(obs_std[j])
        term1 = _kernel_term(obs[j] - mu, sig).mean()         # E|F - y|
        diff = mu[:, None] - mu[None, :]                      # (n, n)
        term2 = 0.5 * _kernel_term(diff, sqrt(2.0) * sig).mean()  # 0.5 E|F - F'|
        crps[j] = term1 - term2
    return crps


if __name__ == "__main__":
    obs, variances = load_calibration_obs(OBS)
    cell_idx = load_cell_indices(CELLS, obs.size)
    obs_x = cell_idx / N_FIELD

    chain = load_chain_params(CHAIN)
    fields = surrogate_fields(chain, T_FINAL, SURROGATE)
    pred = fields[:, cell_idx]

    pmean = pred.mean(axis=0)
    plo, pmed, phi = np.percentile(pred, [2.5, 50, 97.5], axis=0)
    covered = (obs >= plo) & (obs <= phi)

    # Full-domain predictive median + 95% band (every cell, not just obs cells).
    flo, fmed, fhi = np.percentile(fields, [2.5, 50, 97.5], axis=0)

    obs_std = np.sqrt(variances) if variances.size == obs.size else None
    crps = crps_ensemble(pred, obs, obs_std=obs_std)

    print(f"[obs]    {OBS}: m = {obs.size} cells {cell_idx.tolist()}")
    print(f"[cells]  {CELLS}")
    print(f"[chain]  {CHAIN}: {chain.shape[0]} draws evaluated "
          f"(cap {MAX_SAMPLES})")
    print(f"[surrogate] {SURROGATE}")
    print(f"[result] pred shape = {pred.shape}")
    print(f"[result] surrogate-pred mean RMSE vs observed = "
          f"{np.sqrt(np.mean((pmean - obs) ** 2)):.4e}")
    print(f"[result] observed value inside 95% predictive band at "
          f"{covered.sum()}/{covered.size} obs points")
    obs_err_note = (f"obs error folded in (std {obs_std.min():.2e}–{obs_std.max():.2e})"
                    if obs_std is not None else "no obs error")
    print(f"[crps] mean CRPS vs observed = {crps.mean():.4e} "
          f"(lower is better; density units; {obs_err_note})")
    print(f"[crps] per-cell min/max = {crps.min():.4e} / {crps.max():.4e}")

    np.savetxt(
        CRPS_DAT,
        np.column_stack([cell_idx, obs_x, obs, pmed, crps]),
        fmt=["%d", "%.6f", "%.8e", "%.8e", "%.8e"],
        header=f"per-cell CRPS  (mean {crps.mean():.6e}, {chain.shape[0]} draws, "
               f"t={T_FINAL:.4e})\ncell_idx  obs_x  obs  surrogate_median  crps",
    )
    print(f"[crps] wrote per-cell scores to {CRPS_DAT}")

    if NO_PLOT:
        raise SystemExit(0)

    from plots import plot_crps, plot_field_violins, plot_obs_histograms

    # Exact Sod overlay at the chain median (obs truth is not stored in sod_obs.dat).
    exact_rho = exact_density_on_cells(np.median(chain, axis=0), T_FINAL, N_FIELD)
    obs_every = int(np.median(np.diff(cell_idx))) if cell_idx.size > 1 else 15


    plot_field_violins(OUT, obs_x=obs_x, pred=pred, obs=obs, pmed=pmed,
                       exact_rho=exact_rho, n_field=N_FIELD,
                       obs_every=obs_every, t=T_FINAL, n_draws=chain.shape[0])
    plot_obs_histograms(OUT.with_name(OUT.stem + "_hist" + OUT.suffix),
                        cell_idx=cell_idx, obs_x=obs_x, pred=pred, obs=obs,
                        pmed=pmed, plo=plo, phi=phi, t=T_FINAL,
                        n_draws=chain.shape[0])
    plot_crps(OUT.with_name(OUT.stem + "_crps" + OUT.suffix), obs_x=obs_x,
              cell_idx=cell_idx, crps=crps, exact_rho=exact_rho, pmed=pmed,
              fmed=fmed, flo=flo, fhi=fhi, obs=obs, n_field=N_FIELD,
              t=T_FINAL, chain_name=CHAIN.name)
