import argparse
import os
import pickle
import sys
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

import h5py
import numpy as np
from sklearn.metrics import mean_squared_error

from helpers.log import get_logger
from helpers.resources import record_resources
from workflows.run_context import SubRunContext
from task_simulations.CDR_1D.sim_wrapper import run_cdr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOCK_TUBE_VALIDATION_DIR = ROOT / "training_data" / "shock_tube" / "validation_set"
N_VAL = 50

# Loaded once per worker process (see _init_mse_worker).
_worker_surrogate = None
_worker_space = None


def _init_mse_worker(surrogate_file: str, param_space) -> None:
    global _worker_surrogate, _worker_space
    with open(surrogate_file, "rb") as f:
        _worker_surrogate = pickle.load(f)
    _worker_space = param_space


def _process_validation_path(path_str: str) -> tuple[float, int]:
    """Read one validation HDF5, predict, return (sse, count) over all times/sol_keys.

    X and Y are built exactly as in training (``param_space.construct_X/Y``) so
    the prediction and ground truth share the same layout. ``predict`` returns
    physical-space values (it inverse-transforms with ``y_scaler``), where the
    fields differ in scale by ~10^5 — so energy would swamp the error. We map
    both sides back into scaled space with the surrogate's fitted ``y_scaler``
    so every field contributes comparably (this is the space the GP optimizes).
    """
    X = _worker_space.construct_X(path_str)
    Y_true = _worker_space.construct_Y(path_str)
    Y_pred, _ = _worker_surrogate.predict(X, return_std=False)

    ys = _worker_surrogate.y_scaler
    diff = ys.transform(np.asarray(Y_pred)) - ys.transform(Y_true)
    return float(np.sum(diff * diff)), int(diff.size)


def check_mse_shock_tube(
    surrogate_file,
    validation_data_dir,
    param_space,
) -> float:
    """Compute global MSE by reading one validation HDF5 file at a time."""

    logger = get_logger(__name__)
    validation_data_dir = Path(validation_data_dir)
    if not validation_data_dir.is_dir():
        logger.error("Validation directory not found: %s", validation_data_dir)
        raise FileNotFoundError(f"Validation directory not found: {validation_data_dir}")

    paths = sorted(validation_data_dir.glob("shock_tube__*.h5"))
    if not paths:
        logger.error("No validation HDF5 files in %s", validation_data_dir)
        raise FileNotFoundError(f"No validation HDF5 files in {validation_data_dir}")

    path_strs = [str(p) for p in paths]
    n_workers = 5

    with mp.Pool(
        processes=n_workers,
        initializer=_init_mse_worker,
        initargs=(str(surrogate_file), param_space),
    ) as pool:
        results = pool.map(_process_validation_path, path_strs)

    sse = sum(r[0] for r in results)
    n = sum(r[1] for r in results)

    mse = sse / n

    logger.info("Global MSE=%.6e", mse)

    return mse


def check_mse_cdr(surrogate_file) -> float:
    """CDR validation MSE (not yet wired to on-disk validation data)."""
    raise NotImplementedError(
        "check_mse_cdr is not implemented; use model_name='shock_tube' or add a CDR validator"
    )


def _mse_shock_tube(*, surrogate_file, validation_data_dir, param_space, **_kwargs) -> float:
    return check_mse_shock_tube(
        surrogate_file=surrogate_file,
        validation_data_dir=validation_data_dir,
        param_space=param_space,
    )


def _mse_cdr(*, surrogate_file, **_kwargs) -> float:
    return check_mse_cdr(surrogate_file=surrogate_file)


MSE_CHECKERS = {
    "shock_tube": _mse_shock_tube,
    "cdr": _mse_cdr,
}


def check_mse(
    surrogate_file,
    model_name,
    validation_data_dir,
    param_space,
) -> float:
    """Compute MSE of the surrogate on a held-out validation set in physical space."""
    try:
        checker = MSE_CHECKERS[model_name]
    except KeyError as exc:
        known = ", ".join(sorted(MSE_CHECKERS))
        raise ValueError(
            f"No MSE checker for model {model_name!r}; known: {known}"
        ) from exc
    return checker(
        surrogate_file=surrogate_file,
        validation_data_dir=validation_data_dir,
        param_space=param_space,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", required=True, help="Path to context.json")
    args = parser.parse_args()

    ctx = SubRunContext.load(args.ctx)
    g, a = ctx.global_run_context, ctx.run_artifacts

    logger = get_logger(__name__)
    with record_resources(logger, "check_mse"):
        mse = check_mse(
            surrogate_file=a.surrogate_file,
            model_name=g.model_name,
            validation_data_dir=DEFAULT_SHOCK_TUBE_VALIDATION_DIR,
            param_space=ctx.param_space,
        )
    # NOTE: stdout is consumed by rose as the criterion metric value
    # (parsed via float(task["stdout"])); keep this as the only stdout write.
    print(mse)
