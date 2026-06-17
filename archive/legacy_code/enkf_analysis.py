"""One stochastic (perturbed-observation) EnKF analysis step in numpy.

Mirrors EnKF/EnKF.h so the Python cycled driver uses the same update the C++
prototype does:

    P   = cov(X^f)                      # background covariance (n x n)
    PH  = P H                           # state-obs cross covariance (n x m)
    S   = H^T P H + R                   # innovation covariance (m x m)
    K   = PH S^{-1}                     # Kalman gain (n x m)
    X^a = X^f + K (y^o_e - H^T X^f)     # perturbed-obs update

H follows the convention ``H^T @ state`` maps state -> observation space, i.e.
H has shape (n_state, m) with a single 1 per column on the observed row.

Optional Gaspari-Cohn Schur-product localization tapers spurious long-range
sample covariances. The leading ``num_globals`` (augmented) rows have no spatial
location, so they are left untapered.
"""

from __future__ import annotations

import numpy as np


def gaspari_cohn(dist: np.ndarray, radius: float) -> np.ndarray:
    """5th-order compact-support taper; 1 at dist 0, 0 for |dist| >= radius."""
    if radius <= 0.0:
        return np.ones_like(np.asarray(dist, dtype=float))
    c = radius / 2.0
    r = np.abs(np.asarray(dist, dtype=float)) / c
    w = np.zeros_like(r)
    near = r <= 1.0
    mid = (r > 1.0) & (r < 2.0)
    w[near] = (-0.25 * r[near] ** 5 + 0.5 * r[near] ** 4 + 0.625 * r[near] ** 3
               - (5.0 / 3.0) * r[near] ** 2 + 1.0)
    rm = r[mid]
    w[mid] = ((1.0 / 12.0) * rm ** 5 - 0.5 * rm ** 4 + 0.625 * rm ** 3
              + (5.0 / 3.0) * rm ** 2 - 5.0 * rm + 4.0 - (2.0 / 3.0) / rm)
    return w


def _localization_matrices(state_loc, obs_loc, num_globals, loc_rad):
    """Build C_xy (n x m) and C_yy (m x m) Schur-product taper matrices."""
    state_loc = np.asarray(state_loc, dtype=float)
    obs_loc = np.asarray(obs_loc, dtype=float)
    n, m = state_loc.size, obs_loc.size

    d_xy = np.abs(state_loc[:, None] - obs_loc[None, :])      # (n, m)
    C_xy = gaspari_cohn(d_xy, loc_rad)
    C_xy[:num_globals, :] = 1.0                               # globals untapered

    d_yy = np.abs(obs_loc[:, None] - obs_loc[None, :])        # (m, m)
    C_yy = gaspari_cohn(d_yy, loc_rad)
    return C_xy, C_yy


def enkf_analysis(
    X: np.ndarray,
    obs: np.ndarray,
    H: np.ndarray,
    obs_error: float,
    *,
    rng: np.random.Generator | None = None,
    state_loc=None,
    obs_loc=None,
    num_globals: int = 0,
    loc_rad: float = 0.0,
    localize: bool = False,
) -> np.ndarray:
    """Return the analysis ensemble X^a (n x Ne) for forecast ensemble X.

    Parameters
    ----------
    X : (n_state, Ne)
        Forecast ensemble, one member per column.
    obs : (m,)
        Observation vector y^o (unperturbed; this routine perturbs internally).
    H : (n_state, m)
        Observation operator (H^T @ state -> obs space).
    obs_error : float
        Observation noise std sigma; R = sigma^2 I.
    localize : bool
        If True, apply Gaspari-Cohn Schur-product localization using
        ``state_loc``/``obs_loc`` (cell positions), ``num_globals``, ``loc_rad``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, ne = X.shape
    m = obs.size

    Xbar = X.mean(axis=1, keepdims=True)
    A = X - Xbar
    P = (A @ A.T) / (ne - 1.0)                    # (n, n) background covariance

    PH = P @ H                                    # (n, m)
    HPH = H.T @ PH                                # (m, m)

    if localize:
        C_xy, C_yy = _localization_matrices(state_loc, obs_loc, num_globals, loc_rad)
        PH = PH * C_xy
        HPH = HPH * C_yy

    R = (obs_error ** 2) * np.eye(m)
    S = HPH + R
    K = PH @ np.linalg.pinv(S)                    # (n, m)

    # Perturbed observations y^o_e = y^o + eps, eps ~ N(0, R), per member.
    obs_pert = obs[:, None] + obs_error * rng.standard_normal((m, ne))
    innovation = obs_pert - H.T @ X               # (m, Ne)

    return X + K @ innovation
