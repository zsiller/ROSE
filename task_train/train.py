import os
import pickle
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

from helpers.log import get_logger
from helpers.resources import record_resources
from task_train.model import Surrogate
from workflows.run_context import SubRunContext
from workflows.parameter_spaces import ParameterSpace

# Default number of training calls between full GP hyperparameter
# re-optimizations; other iterations warm-restart with frozen hyperparameters for
# speed. Overridden per-sub by Config.reoptimize_every (context.json).

REOPTIMIZE_EVERY = 25

def train(
    data_path: str,
    surrogate_file: str,
    pod_inc: bool,
    pod_n_components: int,
    param_space: ParameterSpace,
    reoptimize_every: int = REOPTIMIZE_EVERY,
):
    logger = get_logger(__name__)

    warm_start = os.path.exists(surrogate_file)
    if warm_start:
        with open(surrogate_file, "rb") as f:
            surrogate = pickle.load(f)
        logger.info("Loaded existing surrogate with kernel: %s", surrogate.kernel)
    else:
        surrogate = Surrogate(n_pod_components=pod_n_components, pod_inc=pod_inc)
        logger.info("Creating new surrogate with kernel: %s", surrogate.kernel)

    # Cold start always optimizes; warm iterations reuse hyperparameters except
    # every REOPTIMIZE_EVERY trainings, when we re-optimize as the domain fills.
    prior_trainings = getattr(surrogate, "n_trainings", 0)
    reoptimize = (not warm_start) or (prior_trainings % reoptimize_every == 0)

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    for file in sorted(
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.endswith(".h5")
    ):
        X = param_space.construct_X(file)
        Y = param_space.construct_Y(file)
        X_list.append(X)
        Y_list.append(Y)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    surrogate.train(X, Y, reoptimize=reoptimize)
    logger.info(
        "Surrogate trained on %s: %d rows from %d file(s) (history), "
        "warm_start=%s, reoptimize=%s",
        param_space.name,
        X.shape[0],
        len(X_list),
        warm_start,
        reoptimize,
    )

    with open(surrogate_file, "wb") as f:
        pickle.dump(surrogate, f)

    return {
        "surrogate_file": surrogate_file,
        "pod_inc": pod_inc,
        "pod_n_components": pod_n_components,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", required=True, help="Path to context.json")
    args = parser.parse_args()

    ctx = SubRunContext.load(args.ctx)
    g, c, a = ctx.global_run_context, ctx.run_config, ctx.run_artifacts
    logger = get_logger(__name__)

    with record_resources(logger, "training"):
        train(
            data_path=g.data_dir,
            surrogate_file=a.surrogate_file,
            pod_inc=c.pod_inc,
            pod_n_components=c.pod_n_components,
            param_space=ctx.param_space,
            reoptimize_every=c.reoptimize_every,
        )
