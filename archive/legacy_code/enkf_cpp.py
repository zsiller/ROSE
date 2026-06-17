"""Python bridge to the C++ root filter (EnKF/enkf_step, built from EnKF.h).

The cycle driver builds the ensemble, observations and operator in Python, then
hands the matrix-heavy analysis to the compiled C++ EnKF via a small binary
bundle. This keeps the per-step Kalman update in Eigen while Python manages the
loop. The bundle layout matches EnKF/enkf_step.cpp (column-major, little-endian).
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "EnKF" / "enkf_step.cpp"
_BIN = _ROOT / "EnKF" / "enkf_step"
_EIGEN_INC = "/usr/include/eigen3"


def _ensure_built() -> Path:
    """Compile EnKF/enkf_step if the binary is missing or older than its source."""
    if _BIN.exists() and _BIN.stat().st_mtime >= _SRC.stat().st_mtime:
        return _BIN
    cmd = ["g++", "-O3", "-std=c++17", f"-I{_EIGEN_INC}", str(_SRC), "-o", str(_BIN)]
    subprocess.run(cmd, check=True)
    return _BIN


def enkf_filter_cpp(
    X: np.ndarray,
    obs: np.ndarray,
    H: np.ndarray,
    obs_error: float,
    *,
    state_loc: np.ndarray,
    obs_loc: np.ndarray,
    num_globals: int,
    loc_rad: float,
    localize: bool,
    seed: int = -1,
) -> np.ndarray:
    """Run one C++ EnKF analysis step and return the analysis ensemble X^a.

    Signature mirrors enkf_analysis.enkf_analysis so the two are drop-in
    interchangeable from the cycle driver.
    """
    binary = _ensure_built()

    X = np.ascontiguousarray(X, dtype=np.float64)
    H = np.ascontiguousarray(H, dtype=np.float64)
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    state_loc = np.ascontiguousarray(state_loc, dtype=np.float64)
    obs_loc = np.ascontiguousarray(obs_loc, dtype=np.float64)

    n_state, ne = X.shape
    m = obs.size

    header = struct.pack(
        "<5iqdd", n_state, ne, m, int(num_globals), int(bool(localize)),
        int(seed), float(obs_error), float(loc_rad),
    )

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.bin"
        out_path = Path(td) / "out.bin"
        with open(in_path, "wb") as f:
            f.write(header)
            f.write(X.tobytes(order="F"))           # column-major for Eigen
            f.write(obs.tobytes())
            f.write(H.tobytes(order="F"))
            f.write(state_loc.tobytes())
            f.write(obs_loc.tobytes())

        subprocess.run([str(binary), str(in_path), str(out_path)], check=True)

        data = np.fromfile(out_path, dtype=np.float64, count=n_state * ne)

    return data.reshape((n_state, ne), order="F")
