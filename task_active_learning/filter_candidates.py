"""Filter new candidates based on sample history."""

from __future__ import annotations

import os
import pickle

import numpy as np

from helpers.log import get_logger


def _is_duplicate(candidate: np.ndarray, history: np.ndarray, tol: float = 0.0) -> bool:
    if history.size == 0:
        return False
    if tol <= 0:
        return np.any(np.all(np.isclose(history, candidate, rtol=0, atol=0), axis=1))
    return np.any(np.all(np.isclose(history, candidate, rtol=tol, atol=0), axis=1))


def filter_new_candidates(candidates: np.ndarray, sample_history_path: str, tol: float = 0.0) -> np.ndarray:
    if sample_history_path is None:
        return candidates
    
    if os.path.exists(sample_history_path):
        with open(sample_history_path, "rb") as f:
            history = pickle.load(f)
    else:
        logger = get_logger(__name__)
        logger.error("Sample history file not found: %s", sample_history_path)
        raise FileNotFoundError(f"Sample history file not found: {sample_history_path}")
    
    if history.size == 0:
        return candidates

    kept: list[np.ndarray] = []
    for ic in candidates:
        if not _is_duplicate(ic, history, tol=tol):
            kept.append(ic)
    return np.asarray(kept, dtype=float) if kept else np.empty((0, 4), dtype=float)

