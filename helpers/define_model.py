import pickle
import sys

# Some surrogate.pkl files (e.g. run_200) were pickled when the surrogate package
# was named `surrogate`; the classes now live under `task_train`. Alias the old
# module names so pickle.load can resolve `surrogate.model` / `surrogate.POD`.
import task_train
import task_train.model
import task_train.POD

sys.modules.setdefault("surrogate", task_train)
sys.modules.setdefault("surrogate.model", task_train.model)
sys.modules.setdefault("surrogate.POD", task_train.POD)

SURROGATE_FILE = "/home/zhsiller/research/ROSE/training_runs/shock_tube/run_200/wf_0/surrogate.pkl"

if __name__ == "__main__":
    with open(SURROGATE_FILE, "rb") as f:
        surrogate = pickle.load(f)

    model = surrogate.gp

    params = model.kernel_.get_params()
    print("Learned kernel hyperparameters:")
    for name, value in params.items():
        print(f"  {name}: {value}")