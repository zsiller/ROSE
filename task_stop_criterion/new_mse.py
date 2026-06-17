import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

from sklearn.metrics import mean_squared_error

from helpers.log import get_logger
from helpers.resources import record_resources
from workflows.run_context import SubRunContext

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOCK_TUBE_VALIDATION_DIR = ROOT / "training_data" / "shock_tube" / "validation_set"

def get_mse(surrogate, file_path, param_space):
    """MSE for one validation file, in the surrogate's scaled space.

    ``predict`` returns physical units (it inverse-transforms with ``y_scaler``),
    where energy (~1e5) would swamp rho (~1). Re-apply ``y_scaler`` to both sides
    so every field contributes comparably.
    """
    X_test = param_space.construct_X(file_path)
    Y_pred, _ = surrogate.predict(X_test, return_std=False)
    Y_true = param_space.construct_Y(file_path)
    ys = surrogate.y_scaler
    return mean_squared_error(ys.transform(Y_true), ys.transform(Y_pred))

def check_mse(surrogate_file, validation_data_dir, param_space):
    if not Path(surrogate_file).exists():
        raise FileNotFoundError(f"Surrogate file not found: {surrogate_file}")
    with open(surrogate_file, "rb") as f:
        surrogate = pickle.load(f)
    files = sorted(validation_data_dir.glob("shock_tube__*.h5"))
    if not files:
        raise FileNotFoundError(f"No validation HDF5 files in {validation_data_dir}")
    global_mse = sum(get_mse(surrogate, file, param_space) for file in files)
    return global_mse / len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", required=True, help="Path to context.json")
    args = parser.parse_args()

    ctx = SubRunContext.load(args.ctx)
    a = ctx.run_artifacts

    logger = get_logger(__name__)
    with record_resources(logger, "check_mse"):
        mse = check_mse(
            surrogate_file=a.surrogate_file,
            validation_data_dir=DEFAULT_SHOCK_TUBE_VALIDATION_DIR,
            param_space=ctx.param_space
        )
    print(mse)