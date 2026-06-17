"""
Utilities for reading 1D CDR simulation output written by `run_cdr_1d.cpp`.

The C++ writer stores:
- Root attributes:
  - int    grid_s
  - int    n_steps
  - int    steps_between_snapshots
  - double dt
  - double t_final
  - double beta
  - double a
  - double mu
- Datasets:
  - "x": 1D cell-center locations (length = grid_s)
  - "t": 1D snapshot times (length = n_steps)
  - "u": 2D solution (n_steps x grid_s); u[j, i] = value at time t[j], position x[i]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd


@dataclass
class CDR1DMetadata:
    grid_s: int
    n_steps: int
    steps_between_snapshots: int
    dt: float
    t_final: float
    beta: float
    a: float
    mu: float


def read_cdr_1d_h5(path: str | Path) -> Tuple[pd.DataFrame, CDR1DMetadata]:
    """
    Read the HDF5 file produced by `run_cdr_1d.cpp` into a pandas DataFrame.

    Parameters
    ----------
    path:
        Filesystem path to the HDF5 file, e.g. "cdr_1d_output.h5".

    Returns
    -------
    df : pd.DataFrame
        Rows = x (cell-center locations), columns = snapshot times.
        df.loc[x_i, t_j] is the solution at position x_i and time t_j.
    meta : CDR1DMetadata
        Simulation metadata from root attributes.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        x = np.asarray(f["x"], dtype=np.float64)
        t = np.asarray(f["t"], dtype=np.float64)
        u = np.asarray(f["u"], dtype=np.float64)

        if u.ndim != 2:
            raise ValueError(f"Expected 2D 'u' (n_steps x grid_s), got shape {u.shape}")

        n_steps, grid_s = u.shape
        if x.size != grid_s:
            raise ValueError(
                f"Size mismatch: u has grid_s={grid_s}, len(x)={x.size}"
            )
        if t.size != n_steps:
            raise ValueError(
                f"Size mismatch: u has n_steps={n_steps}, len(t)={t.size}"
            )

        grid_s_attr = int(f.attrs.get("grid_s", grid_s))
        if grid_s != grid_s_attr:
            raise ValueError(f"grid_s attribute ({grid_s_attr}) != u shape ({grid_s})")

        meta = CDR1DMetadata(
            grid_s=grid_s,
            n_steps=int(f.attrs.get("n_steps", n_steps)),
            steps_between_snapshots=int(f.attrs.get("steps_between_snapshots", -1)),
            dt=float(f.attrs.get("dt", np.nan)),
            t_final=float(f.attrs.get("t_final", np.nan)),
            beta=float(f.attrs.get("beta", np.nan)),
            a=float(f.attrs.get("a", np.nan)),
            mu=float(f.attrs.get("mu", np.nan)),
        )

        # u[j, i] = value at time t[j], position x[i]
        # DataFrame: rows = x, columns = t  =>  df.iloc[i, j] = u[j, i]
        df = pd.DataFrame(
            u.T,
            index=pd.Index(x, name="x"),
            columns=pd.Index(t, name="t"),
        )

    return df, meta


__all__ = ["CDR1DMetadata", "read_cdr_1d_h5"]
