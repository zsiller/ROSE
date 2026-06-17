import os
import sys
import asyncio

from rose.al.active_learner import SequentialActiveLearner

from radical.asyncflow import WorkflowEngine
from rose.metrics import MEAN_SQUARED_ERROR_MSE


async def rose_al():
    # Enable dry run in the workflow engine
    asyncflow = await WorkflowEngine.create(dry_run=True)

    # Create an active learner with the workflow engine
    acl = SequentialActiveLearner(asyncflow)

    # Path to your training script or code
    code_path = f'{sys.executable} {os.getcwd()}'

    # Now use `acl` to define and simulate the workflow...
    # (e.g., acl.run(...), acl.sample(...), etc.)
    # During dry run, tasks will be logged but not executed.
    @acl.simulation_task
    async def simulation(*args, **kwargs):
        return f'{code_path}/sim.py'

    @acl.training_task
    async def training(*args, **kwargs):
        return f'{code_path}/train.py'

    @acl.active_learn_task
    async def active_learn(*args, **kwargs):
        return f'{code_path}/active_learning.py'

    async for state in acl.start(max_iter=10):
        print(f"Iteration {state.iteration}: {state.metric_value}")     

    await acl.shutdown()

asyncio.run(rose_al())

print("Dry run completed")