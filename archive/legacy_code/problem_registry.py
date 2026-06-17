"""Unified dispatch tables keyed by problem / model name."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# Import SimRunners from task_simulations
from task_simulations.CDR_1D.sim_wrapper import run_cdr
from task_simulations.Shock_Tube.run_shock_tube import run_shock_tube

SimRunner = Callable[..., tuple[dict, dict]]
SurrogateXBuilder = Callable[[np.ndarray, np.ndarray], np.ndarray]
MseChecker = Callable[..., float]


def _shock_tube_surrogate_x(row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ic = row[0]
    return np.column_stack([np.tile(ic, (len(grid), 1)), grid])


def _cdr_surrogate_x(row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    beta = row[0, 0]
    return np.column_stack([np.full(len(grid), beta), grid])


PROBLEM_RUNNERS: dict[str, SimRunner] = {
    "cdr": run_cdr,
    "shock_tube": run_shock_tube,
}

SURROGATE_X_BUILDERS: dict[str, SurrogateXBuilder] = {
    "shock_tube": _shock_tube_surrogate_x,
    "cdr": _cdr_surrogate_x,
}

# Populated by task_stop_criterion.check_mse after checker functions are defined.
MSE_CHECKERS: dict[str, MseChecker] = {}


def known_keys(table: dict[str, Any]) -> str:
    return ", ".join(sorted(table))


def dispatch_problem(problem: str, output_path: str, params: list[float]) -> tuple[dict, dict]:
    try:
        return PROBLEM_RUNNERS[problem](output_path, *params)
    except KeyError as exc:
        raise ValueError(
            f"Unknown problem {problem!r}; known: {known_keys(PROBLEM_RUNNERS)}"
        ) from exc


def dispatch_surrogate_x(name: str, row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        return SURROGATE_X_BUILDERS[name](row, grid)
    except KeyError as exc:
        raise ValueError(
            f"Unknown parameter space: {name!r}; known: {known_keys(SURROGATE_X_BUILDERS)}"
        ) from exc


def dispatch_mse(model_name: str, **kwargs: Any) -> float:
    try:
        return MSE_CHECKERS[model_name](**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unknown model_name {model_name!r}; known: {known_keys(MSE_CHECKERS)}"
        ) from exc
