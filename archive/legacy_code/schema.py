"""On-disk artifacts and shared run parameters for pipeline subprocesses."""

from __future__ import annotations

import dataclasses
import os


@dataclasses.dataclass
class SimData:
    Y_path: str = "Y.pkl"
    X_path: str = "X.pkl"
    new_betas_path: str = "new_betas.json"
    beta_history_path: str = "beta_history.json"


@dataclasses.dataclass
class TrainData:
    surrogate_file: str = "surrogate.pkl"
    pod_file: str = "pod.pkl"
    type: str = "uncertainty"


@dataclasses.dataclass
class ValidationData:
    """Held-out validation tensors; kept under figure/outputs so they persist across clean()."""

    val_x_path: str = "figure/outputs/validation_set/validation_inputs.pkl"
    val_y_path: str = "figure/outputs/validation_set/validation_targets.pkl"


# Shared physics / simulation settings (keep in sync across sim, AL, and validation).
BETA_RANGE: tuple[float, float] = (0.0, 10.0)
T_FINAL: float = 0.25
WRITE_EVERY: int = 1000


def all_artifact_paths() -> list[str]:
    """Default filenames produced or consumed by the pipeline (for cleanup).

    Validation pickles under figure/outputs are intentionally omitted so a fixed
    held-out set is not deleted by ``clean()``.
    """
    s = SimData()
    t = TrainData()
    return [
        s.Y_path,
        s.X_path,
        s.new_betas_path,
        s.beta_history_path,
        t.surrogate_file,
        t.pod_file,
    ]


def clean() -> None:
    """Remove all persisted state files from previous runs."""
    for path in all_artifact_paths():
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")


def save(result_dir: str = "temp"):
    """
    Moves persisted artifacts to the results/output folder
    so they are preserved, i.e., not deleted by `clean()`.
    """
    import shutil

    path = "results/outputs/" + result_dir

    s = SimData()
    t = TrainData()
    os.makedirs(path, exist_ok=True)

    artifact_paths = [
        (s.Y_path, os.path.join(path, os.path.basename(s.Y_path))),
        (s.X_path, os.path.join(path, os.path.basename(s.X_path))),
        (s.beta_history_path, os.path.join(path, os.path.basename(s.beta_history_path))),
        (t.surrogate_file, os.path.join(path, os.path.basename(t.surrogate_file))),
        (t.pod_file, os.path.join(path, os.path.basename(t.pod_file))),
    ]

    for src, dst in artifact_paths:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")


    
