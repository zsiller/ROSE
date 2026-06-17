import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.active_learner import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from workflows.schema import clean, save

from helpers.timing import Timer

AL_CYCLES = 11

async def active_learning_workflow():

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())

    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    code_path = f"{sys.executable} {os.getcwd()}"

    @acl.simulation_task
    async def simulation(*args):
        return f'{code_path}/simulations/sim.py'

    @acl.training_task
    async def training(*args):
        return f'{code_path}/surrogate/train.py'

    @acl.active_learn_task
    async def active_learn(*args):
        return f'{code_path}/active_learning/active_learning.py'

    @acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.0000000001)
    async def check_mse(*args):
        return f'{code_path}/stop_criterion/check_mse.py'

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

    save(result_dir="uncertainty_model_test")