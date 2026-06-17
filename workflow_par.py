import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose import LearnerConfig, TaskConfig
from rose.al import ParallelActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from helpers.log import get_logger
from helpers.timing import Timer
from workflows.run_context import GlobalRunContext, SubRunContext

# Shock-tube surrogate: StandardScaler on X and Y, optional POD, GP on rho fields.
# Each parallel learner gets its own wf_* SubRunContext with a distinct n_select.

SEED = np.array([[100000, 10000, 1.0, 0.125]])


def build_parallel_contexts(
    run_label: str = "run_parallel_2",
    n_selects: tuple[int, ...] = (2, 3, 5),
    *,
    al_method: str = "uncertainty",
    max_iter: int = 10,
    model_name: str = "shock_tube",
    convergence_threshold: float = 1e-4,
    pod_inc: bool = False,
    pod_n_components: int = 20,
) -> list[SubRunContext]:
    """Create one SubRunContext per parallel learner, differing only in n_select."""
    # Creating the campaign context opens {run_dir}/rose.log and logs the
    # global settings; each create_sub() opens its own wf_*/rose.log.
    global_ctx = GlobalRunContext(
        run_label=run_label,
        model_name=model_name,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
    )

    contexts = [
        global_ctx.create_sub(
            wf_ID=f"wf_{i}",
            al_method=al_method,
            pod_inc=pod_inc,
            pod_n_components=pod_n_components,
            n_select=n_select,
        )
        for i, n_select in enumerate(n_selects)
    ]
    for ctx in contexts:
        ctx.gen_seed(SEED)
    return contexts


def learner_config_for(ctx: SubRunContext) -> LearnerConfig:
    """Pin all tasks for one learner to the same context.json."""
    ctx_file = ctx.run_artifacts.context_file
    task = TaskConfig(kwargs={"--ctx": ctx_file})
    return LearnerConfig(
        simulation=task,
        training=task,
        active_learn=task,
        criterion=task,
    )


async def active_learning_parallel(contexts: list[SubRunContext]) -> None:
    logger = get_logger(__name__)
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    al = ParallelActiveLearner(asyncflow)

    g = contexts[0].global_run_context
    code_path = f"{sys.executable} {os.getcwd()}"

    @al.simulation_task
    async def simulation(*args, **kwargs):
        ctx_file = kwargs["--ctx"]
        return f"{code_path}/task_simulations/sim.py --ctx {ctx_file}"

    @al.training_task
    async def training(*args, **kwargs):
        ctx_file = kwargs["--ctx"]
        return f"{code_path}/task_train/train.py --ctx {ctx_file}"

    @al.active_learn_task
    async def active_learn(*args, **kwargs):
        ctx_file = kwargs["--ctx"]
        return f"{code_path}/task_active_learning/active_learning.py --ctx {ctx_file}"

    @al.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE,
        threshold=g.convergence_threshold,
    )
    async def check_mse(*args, **kwargs):
        ctx_file = kwargs["--ctx"]
        return f"{code_path}/task_stop_criterion/check_mse.py --ctx {ctx_file}"

    learner_configs = [learner_config_for(ctx) for ctx in contexts]

    async for state in al.start(
        parallel_learners=len(contexts),
        max_iter=g.max_iter,
        learner_configs=learner_configs,
    ):
        sub = contexts[state.learner_id]
        logger.info(
            "learner=%s wf=%s n_select=%s iter=%s mse=%s",
            state.learner_id,
            sub.run_config.wf_ID,
            sub.run_config.n_select,
            state.iteration,
            state.metric_value,
        )

    await al.shutdown()


if __name__ == "__main__":
    contexts = build_parallel_contexts(
        run_label="run_parallel_1",
        n_selects=(2, 3, 5),
    )

    logger = get_logger("rose.run")
    g = contexts[0].global_run_context
    logger.info("Parallel campaign: %s", g.run_dir)
    for ctx in contexts:
        logger.info(
            "  %s: n_select=%s log=%s ctx=%s",
            ctx.run_config.wf_ID,
            ctx.run_config.n_select,
            ctx.run_artifacts.log_file,
            ctx.run_artifacts.context_file,
        )

    asyncio.run(active_learning_parallel(contexts))
    # timer.exit()
    # logger.info("Parallel workflow completed in %s seconds", timer.elapsed())
