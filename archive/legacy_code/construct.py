import h5py
import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


def list_h5_filenames(data_dir: str) -> list[str]:
    """Sorted shock-tube HDF5 absolute filenames under ``data_dir``."""
    data_dir_path = os.path.abspath(data_dir)
    return sorted(
        [
            os.path.join(data_dir_path, f)
            for f in os.listdir(data_dir_path)
            if f.endswith(".h5")
        ]
    )


def _rows_from_h5(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build X (61, 5) and Y (61, 256) from one simulation file."""
    with h5py.File(file_path, "r") as f:
        rho = f["rho"][:]
        time = np.asarray(f["t"], dtype=float)
        p_high = f.attrs.get("p_high")
        p_low = f.attrs.get("p_low")
        rho_high = f.attrs.get("rho_high")
        rho_low = f.attrs.get("rho_low")

    meta_cols = np.column_stack(
        [
            np.full_like(time, fill_value=p_high, dtype=float),
            np.full_like(time, fill_value=p_low, dtype=float),
            np.full_like(time, fill_value=rho_high, dtype=float),
            np.full_like(time, fill_value=rho_low, dtype=float),
            time,
        ]
    )
    return meta_cols, rho


def construct_X_Y_from_files(
    data_dir: str,
    filenames: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training rows only from the given HDF5 filenames."""
    if not filenames:
        raise ValueError("No HDF5 files provided")

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    root = Path(data_dir)

    for name in filenames:
        X_part, Y_part = _rows_from_h5(str(root / name))
        X_list.append(X_part)
        Y_list.append(Y_part)

    return np.vstack(X_list), np.vstack(Y_list)


def construct_X_Y(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build X, Y from every ``.h5`` file in ``data_dir``."""
    return construct_X_Y_from_files(data_dir, list_h5_filenames(data_dir))
