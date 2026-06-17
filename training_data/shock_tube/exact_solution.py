"""Load exact shock-tube rho fields from validation HDF5 and Riemann transfer files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np

DEFAULT_VALIDATION_DIR = Path(__file__).resolve().parent / "validation_data"
DEFAULT_TRANSFER_DIR = Path(__file__).resolve().parent
DEFAULT_TRANSFER_BIN = DEFAULT_TRANSFER_DIR / "shock_tube_transfer"
PARAM_KEYS = ("p_high", "p_low", "rho_high", "rho_low")
N_CELLS = 256
X0 = 0.5


def _params_match(attrs, target: np.ndarray, *, pressure_atol: float = 0.5, rho_atol: float = 1e-4) -> bool:
    ic = np.array([float(attrs[k]) for k in PARAM_KEYS], dtype=float)
    target = np.asarray(target, dtype=float).reshape(4)
    atol = np.array([pressure_atol, pressure_atol, rho_atol, rho_atol])
    return np.all(np.isclose(ic, target, rtol=0.0, atol=atol))


def find_validation_file(
    sim_params: np.ndarray,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
) -> Path | None:
    """Return the newest validation file matching 4D IC params, if any."""
    validation_dir = Path(validation_dir)
    if not validation_dir.is_dir():
        return None

    sim_params = np.asarray(sim_params, dtype=float).reshape(4)
    matches: list[Path] = []
    for path in validation_dir.glob("shock_tube__*.h5"):
        try:
            with h5py.File(path, "r") as f:
                if _params_match(f.attrs, sim_params):
                    matches.append(path)
        except OSError:
            continue

    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_exact_rho(
    point: np.ndarray,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
) -> tuple[np.ndarray, Path] | tuple[None, None]:
    """
    Load exact rho(x) for a 5D query point ``[p_high, p_low, rho_high, rho_low, t]``.

    Returns
    -------
    rho, path
        rho has shape (256,). ``(None, None)`` if no matching validation run exists.
    """
    point = np.asarray(point, dtype=float).reshape(5)
    path = find_validation_file(point[:4], validation_dir)
    if path is None:
        return None, None

    with h5py.File(path, "r") as f:
        times = np.asarray(f["t"], dtype=float)
        rho_series = np.asarray(f["rho"], dtype=float)
        t_idx = int(np.argmin(np.abs(times - point[4])))
        return rho_series[t_idx], path


def load_exact_rho_time_sweep(
    ic: np.ndarray,
    times: np.ndarray,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
) -> tuple[np.ndarray, Path] | tuple[None, None]:
    """Load exact rho(x, t) on ``times`` for fixed 4D IC params."""
    ic = np.asarray(ic, dtype=float).reshape(4)
    times = np.asarray(times, dtype=float)
    path = find_validation_file(ic, validation_dir)
    if path is None:
        return None, None

    with h5py.File(path, "r") as f:
        file_times = np.asarray(f["t"], dtype=float)
        rho_series = np.asarray(f["rho"], dtype=float)
        rows = [rho_series[int(np.argmin(np.abs(file_times - t)))] for t in times]
        return np.stack(rows, axis=0), path


def transfer_file_prefix(sim_params: np.ndarray) -> str:
    sim_params = np.asarray(sim_params, dtype=float).reshape(4)
    params_str = "_".join(f"{round(p, 5)}" for p in sim_params)
    return f"shock_tube__{params_str}"


def transfer_density_path(
    sim_params: np.ndarray,
    transfer_dir: str | Path = DEFAULT_TRANSFER_DIR,
) -> Path:
    prefix = transfer_file_prefix(sim_params)
    return Path(transfer_dir) / f"{prefix}_density.txt"


def load_density_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0], data[:, 1]


def resample_to_cells(x_src: np.ndarray, rho_src: np.ndarray, n_cells: int = N_CELLS) -> np.ndarray:
    x_cells = (np.arange(n_cells, dtype=float) + 0.5) / n_cells
    return np.interp(x_cells, x_src, rho_src)


def ensure_transfer_density(
    point: np.ndarray,
    transfer_dir: str | Path = DEFAULT_TRANSFER_DIR,
    transfer_bin: str | Path = DEFAULT_TRANSFER_BIN,
) -> Path:
    point = np.asarray(point, dtype=float).reshape(5)
    out_path = transfer_density_path(point[:4], transfer_dir)
    if out_path.is_file():
        return out_path

    transfer_bin = Path(transfer_bin)
    if not transfer_bin.is_file():
        raise FileNotFoundError(f"Transfer binary not found: {transfer_bin}")

    prefix = transfer_file_prefix(point[:4])
    subprocess.run(
        [
            str(transfer_bin),
            prefix,
            str(point[0]),
            str(point[1]),
            str(point[2]),
            str(point[3]),
            str(point[4]),
        ],
        check=True,
        cwd=str(transfer_dir),
    )
    if not out_path.is_file():
        raise FileNotFoundError(f"Transfer output not created: {out_path}")
    return out_path


def load_transfer_rho(
    point: np.ndarray,
    transfer_dir: str | Path = DEFAULT_TRANSFER_DIR,
    transfer_bin: str | Path = DEFAULT_TRANSFER_BIN,
    *,
    n_cells: int = N_CELLS,
) -> tuple[np.ndarray, Path]:
    """Load Riemann exact rho(x) from shock_tube_transfer output, resampled to cell grid."""
    path = ensure_transfer_density(point, transfer_dir, transfer_bin)
    x_src, rho_src = load_density_txt(path)
    return resample_to_cells(x_src, rho_src, n_cells), path


def exact_high_low_densities(rho_high: float, rho_low: float, n_cells: int = N_CELLS) -> tuple[np.ndarray, np.ndarray]:
    """Return uniform high- and low-density reference profiles on the cell grid."""
    high = np.full(n_cells, float(rho_high), dtype=float)
    low = np.full(n_cells, float(rho_low), dtype=float)
    return high, low
