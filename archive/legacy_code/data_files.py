"""Match ``new_sample.pkl`` IC rows to HDF5 files written by ``simulations/sim.py``."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np

from task_simulations.sim_interface import params_key


def find_h5_for_sample(
    data_dir: str,
    sample: np.ndarray,
    *,
    model_name: str = "shock_tube",
) -> str | None:
    """Return the newest HDF5 in ``data_dir`` matching this IC (by param key)."""
    key = params_key(sample)
    prefix = f"{model_name}__{key}__"
    data_path = Path(data_dir)
    matches = [
        p.name
        for p in data_path.iterdir()
        if p.suffix == ".h5" and p.name.startswith(prefix)
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda name: os.path.getmtime(data_path / name),
    )


def h5_files_for_new_samples(
    data_dir: str,
    new_sample_path: str,
    *,
    model_name: str = "shock_tube",
    logger: logging.Logger | None = None,
) -> list[str]:
    """Resolve HDF5 filenames for ICs listed in ``new_sample.pkl``."""
    log = logger or logging.getLogger(__name__)
    with open(new_sample_path, "rb") as f:
        samples = np.asarray(pickle.load(f), dtype=float)

    if samples.size == 0:
        raise ValueError(f"Empty sample file: {new_sample_path}")

    samples = np.atleast_2d(samples)
    files: list[str] = []
    for i, sample in enumerate(samples):
        name = find_h5_for_sample(data_dir, sample, model_name=model_name)
        if name is None:
            log.warning(
                "No HDF5 in %s for new_sample row %d (params=%s)",
                data_dir,
                i,
                params_key(sample),
            )
            continue
        files.append(name)

    return files


def h5_files_for_sample_history(
    data_dir: str,
    sample_history_path: str,
    *,
    model_name: str = "shock_tube",
    logger: logging.Logger | None = None,
) -> list[str]:
    """Resolve one HDF5 per unique IC in accumulated ``sample_history.pkl``."""
    from task_active_learning.shock_tube_uncertainty import load_sample_history

    log = logger or logging.getLogger(__name__)
    history = load_sample_history(sample_history_path)
    if history.size == 0:
        return []

    history = np.atleast_2d(history)
    seen: set[str] = set()
    files: list[str] = []
    for i, sample in enumerate(history):
        ic = np.asarray(sample, dtype=float).ravel()[:4]
        key = params_key(ic)
        if key in seen:
            continue
        seen.add(key)
        name = find_h5_for_sample(data_dir, ic, model_name=model_name)
        if name is None:
            log.warning(
                "No HDF5 in %s for sample_history row %d (params=%s)",
                data_dir,
                i,
                key,
            )
            continue
        files.append(name)

    return sorted(files)
