import asyncio
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.active_learner import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from helpers.log import get_logger
from helpers.resources import (
    configure_resources_log,
    log_checkpoint,
    record_resources,
    resources_log_path,
)
from workflows.parameter_spaces import DEFAULT_SPACES, get_parameter_space
from workflows.run_context import GlobalRunContext, SubRunContext



# Shock-tube surrogate: StandardScaler on X and Y, POD on scaled rho, GP on coeffs.
# Training / prediction logic lives in surrogate/model.py (same pattern as hdf5_to_XY.py).


# def default_run_context(
#     al_method: str = "uncertainty",
#     max_iter: int = 3,
#     model_name: str = "shock_tube",
#     convergence_threshold: float = 1e-5,
#     run_label: str = "run_test",
#     wf_ID: str = "wf_0",
#     pod_inc: bool = False,
#     pod_n_components: int = 20,
#     n_select: int = 10,
# ) -> SubRunContext:
#     """Build a sub-context with global params injected via :class:`GlobalRunContext`."""
#     # Creating the campaign context opens {run_dir}/rose.log and logs the
#     # global settings; create_sub() opens its own wf_*/rose.log.
#     global_ctx = GlobalRunContext(
#         run_label=run_label,
#         model_name=model_name,
#         max_iter=max_iter,
#         convergence_threshold=convergence_threshold,
#     )

#     return global_ctx.create_sub(
#         wf_ID=wf_ID,
#         al_method=al_method,
#         pod_inc=pod_inc,
#         pod_n_components=pod_n_components,
#         n_select=n_select,
#     )


async def active_learning_workflow(ctx: SubRunContext) -> None:
    logger = get_logger(__name__)
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    g, a, c = ctx.global_run_context, ctx.run_artifacts, ctx.run_config

    code_path = f"{sys.executable} {os.getcwd()}"
    ctx_flag = f" --ctx {a.context_file}"

    @acl.simulation_task
    async def simulation(*args):
        return f"{code_path}/task_simulations/sim.py{ctx_flag}"

    @acl.training_task
    async def training(*args):
        return f"{code_path}/task_train/train.py{ctx_flag}"

    @acl.active_learn_task
    async def active_learn(*args):
        return f"{code_path}/task_active_learning/active_learning.py{ctx_flag}"

    @acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=g.convergence_threshold)
    async def check_mse(*args):
        return f"{code_path}/task_stop_criterion/new_mse.py{ctx_flag}"

    resources_path = resources_log_path()

    async for state in acl.start(max_iter=g.max_iter):
        logger.info("Iteration %s: mse=%s", state.iteration, state.metric_value)
        log_checkpoint(
            logger,
            "iteration_end",
            iteration=state.iteration,
            jsonl_path=resources_path,
            mse=state.metric_value,
        )

    await acl.shutdown()


if __name__ == "__main__":
    # For a sub-workflow, pick a region (can use narrowed, here for low_p region
    # as example). Pass None for param_space in create_sub to inherit the whole
    # campaign space.
    #     subspace = p_space.narrowed(
    #         p_high=(75000.0, 100000.0), sol_keys=("rho",), name="shock_tube:low_p"
    #     )

    # Canonical operating point, appended to an LHS spread for the seed batch.
    SHOCK_TUBE_DEFAULT_PARAMS = np.array([1.0e5, 1.0e4, 1.0, 0.125])

    p_space = get_parameter_space("shock_tube")
    global_ctx = GlobalRunContext(run_label="run_1000", model_name="shock_tube", max_iter=20)
    sub = global_ctx.create_sub(wf_ID="wf_0", param_space=p_space.narrowed(sol_keys=("rho",)), al_method="uncertainty", pod_inc=False, n_select=10)

    seed = p_space.sample_lhs(4)
    seed = np.vstack([seed, SHOCK_TUBE_DEFAULT_PARAMS])
    sub.gen_seed(seed)

    # Campaign-wide timing -> global logger. (See run_context.py module docstring,
    # "Logging — which logger to use", for picking global_ctx.logger vs sub.logger.)
    logger = global_ctx.logger
    resources_path = configure_resources_log(f"{sub.run_path}/resources.jsonl")
    logger.info("Sub-run path: %s", sub.run_path)
    logger.info("Sub-run log: %s", sub.run_artifacts.log_file)
    logger.info("Resource metrics JSONL: %s", resources_path)
    logger.info("Surrogate pipeline: POD n_components=%d, scaled X/Y", sub.run_config.pod_n_components)

    with record_resources(logger, "workflow_total", jsonl_path=resources_path):
        asyncio.run(active_learning_workflow(sub))
