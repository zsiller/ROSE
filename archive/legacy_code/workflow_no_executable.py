"""Same ROSE active-learning loop as workflow.py, but tasks run in-process (no subprocess scripts)."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.active_learner import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from active_learning.active_learning import active_learning
from helpers.timing import Timer
from simulations.sim import sim
from stop_criterion.check_mse import check_mse as compute_mse
from surrogate.train import train
from workflows.schema import SimData, clean, save

AL_CYCLES = 11


def _consume_new_betas_for_sim():
    """Match ``simulations/sim.py`` __main__: read pending betas and clear the JSON queue."""
    betas = None
    if os.path.exists(SimData.new_betas_path):
        with open(SimData.new_betas_path, "r") as f:
            loaded = json.load(f)
        if loaded:
            betas = loaded
        with open(SimData.new_betas_path, "w") as f:
            json.dump([], f)
    return betas


async def active_learning_workflow():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    @acl.simulation_task(as_executable=False)
    async def simulation(*args):
        betas = _consume_new_betas_for_sim()
        return sim(betas=betas)

    @acl.training_task(as_executable=False)
    async def training(*args):
        return train()

    @acl.active_learn_task(as_executable=False)
    async def active_learn(*args):
        result = active_learning()
        with open(SimData.new_betas_path, "w") as f:
            json.dump(result["new_betas"], f)
        return result

    @acl.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE,
        threshold=0.0000000001,
        as_executable=False,
    )
    async def stop_criterion(*args):
        return compute_mse()

    async for state in acl.start(max_iter=AL_CYCLES):
        print(f"Iteration {state.iteration}: {state.metric_value}")

    await acl.shutdown()


if __name__ == "__main__":
    clean()

    timer = Timer()
    timer.enter()
    print("starting workflow")
    asyncio.run(active_learning_workflow())
    print("workflow completed")
    timer.exit()

    print(f"workflow completed in {timer.elapsed()} seconds")

    save(result_dir="uncertainty_model")
