import pickle
import sys
import argparse
from pathlib import Path
import os

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

from task_active_learning.filter_candidates import filter_new_candidates
from helpers.log import get_logger
from helpers.resources import record_resources
from workflows.run_context import SubRunContext


def active_learning(surrogate, param_space, method, candidate_size, n_select, sample_history_path):
    logger = get_logger(__name__)
    logger.info("Active learning: method=%s, n_select=%s", method, n_select)

    try:
        handler = AL_METHOD_HANDLERS[method]
    except KeyError as exc:
        logger.error("Invalid method: %s", method)
        raise ValueError(f"Invalid method: {method!r}; known: {known}") from exc
    return handler(surrogate, param_space, n_select, candidate_size, sample_history_path)


def uncertainty(surrogate, param_space, n_select, candidate_size, sample_history_path):
    logger = get_logger(__name__)
    logger.info("Uncertainty active learning: n_select=%s, candidate_size=%s", n_select, candidate_size)
    candidates = param_space.sample_lhs(candidate_size)
    candidates = filter_new_candidates(candidates, sample_history_path)

    # Add a column of the final time bound to the end of candidates
    ic = np.hstack([candidates, np.full((candidates.shape[0], 1), param_space.t_bounds[1])])

    if not os.path.exists(surrogate):
        logger.error("Surrogate file not found: %s", surrogate)
        raise FileNotFoundError(f"Surrogate file not found: {surrogate}")

    with open(surrogate, "rb") as f:
        surrogate = pickle.load(f)
    _, std = surrogate.predict(ic, return_std=True)
    std_scalar = np.mean(std, axis=1) if std.ndim > 1 else std
    logger.info("Number of unique uncertainty scores: %s", len(np.unique(std_scalar)))
    top_idx = np.argsort(std_scalar)[-n_select:]
    new_samples = candidates[top_idx]
    logger.info("New samples indices: %s", top_idx)
    logger.info("New samples standard deviations: %s", std_scalar[top_idx])
    return {"new_samples": new_samples}


def random(surrogate, param_space, n_select, candidate_size, sample_history_path):
    logger = get_logger(__name__)
    logger.info("Random active learning: n_select=%s", n_select)
    candidates = param_space.sample_lhs(n_select)
    return filter_new_candidates(candidates, sample_history_path)


AL_METHOD_HANDLERS = {
    "uncertainty": uncertainty,
    "random": random,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", required=True, help="Path to context.json")
    args = parser.parse_args()

    ctx = SubRunContext.load(args.ctx)
    a, c = ctx.run_artifacts, ctx.run_config
    logger = get_logger(__name__)

    with record_resources(logger, "active_learning", method=c.al_method):
        result = active_learning(
            surrogate=a.surrogate_file,
            method=c.al_method,
            n_select=c.n_select,
            candidate_size=c.candidate_size,
            param_space=ctx.param_space,
            sample_history_path=a.sample_history_path,
        )
    with open(a.new_sample_path, "wb") as f:
        pickle.dump(result["new_samples"], f)
    logger.info("New samples written to: %s", a.new_sample_path)


