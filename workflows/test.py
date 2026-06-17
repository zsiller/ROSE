"""Run from project root: ``python workflows/test.py`` (or any cwd; path_setup fixes imports)."""

from pathlib import Path
import sys

# Must run before ``from workflows...`` when this file is executed as a script.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

from workflows.parameter_spaces import DEFAULT_SPACES, get_parameter_space
from workflows.run_context import GlobalRunContext


if __name__ == "__main__":
    get_parameter_space("shock_tube").print()
    print()
    get_parameter_space("cdr").print()

    print(DEFAULT_SPACES["shock_tube"].sample_lhs(10))
    print(DEFAULT_SPACES["cdr"].sample_lhs(10))

    # Narrowing: a sub-workflow exploring only the low-pressure half.
    narrowed = get_parameter_space("shock_tube").narrowed(
        p_high=(75_000.0, 100_000.0), sol_keys=("rho","momentum","energy"), name="shock_tube:low_p",
    )
    narrowed.print()

    global_ctx = GlobalRunContext(run_label="log_test", model_name="shock_tube")
    sub = global_ctx.create_sub(wf_ID="wf_6", param_space=narrowed, al_method="random", pod_inc=True)
    sub.print()
    
    file_path = "/home/zhsiller/research/ROSE/training_runs/shock_tube/run_200/data/shock_tube__100000.0_10000.0_1.0_0.125__wf_0__20260529-080430.h5"
    X = sub.param_space.construct_X(file_path)
    Y = sub.param_space.construct_Y(file_path)
    print(X.shape)
    print(Y.shape)