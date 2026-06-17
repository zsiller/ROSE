"""Run-scoped paths and configuration for active-learning workflows.

Layout::

    {cwd}/training_runs/{model_name}/{run_label}/
        rose.log                 # campaign / driver log (global context)
        data/                    # shared training data (all subs)
        wf_0/
            rose.log             # sub-run log (sim, train, AL, check_mse, …)
            context.json
            new_sample.pkl
            surrogate.pkl
        wf_1/
            ...

``context.json`` is the canonical handoff between the workflow driver and
each subprocess task: drivers persist a resolved snapshot; tasks reconstruct
with :meth:`RunContext.load`.

Logging — which logger to use
-----------------------------
Each context creates and owns a logger already bound to the correct file, so
you rarely call :func:`helpers.log.get_logger` directly. Pick the logger whose
*scope* matches the event:

* ``GlobalRunContext.logger`` → ``{run_dir}/rose.log`` (**campaign scope**).
  Use for anything spanning the whole run: driver start/finish, per-iteration
  progress, total timing, or cross-sub comparisons.
* ``SubRunContext.logger`` → ``{wf_dir}/rose.log`` (**sub scope**). Use for
  events specific to one ``wf_*`` learner.

Rule of thumb: log against the context that owns the event. In a driver you
hold both, so choose per message; campaign-level lines go to the global logger,
sub-specific lines to ``sub.logger``.

Inside the task scripts (``sim.py``, ``train.py``, …) you don't choose at all:
:meth:`SubRunContext.load` pins that sub's log as the active file, so a plain
``get_logger(__name__)`` auto-routes there. See :mod:`helpers.log` for the
handler mechanics.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Allow `python workflows/run_context.py` from the project root.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from workflows.path_setup import ensure_project_root

ensure_project_root(__file__)

from helpers.log import configure_run_log_file, get_logger, log_dataclass
from workflows.parameter_spaces import ParameterSpace, get_parameter_space

__all__ = [
    "Artifact",
    "Config",
    "GlobalRunContext",
    "SubRunContext",
]

_RUNS_ROOT = "training_runs"
_CONTEXT_SCHEMA_VERSION = 1

@dataclass
class GlobalRunContext:
    """Campaign-level settings and shared filesystem paths."""

    run_label: str = "run_1"
    model_name: str = "shock_tube"
    max_iter: int = 11
    convergence_threshold: float = 0.0001

    # Campaign-wide parameter space. Defaults to the registry entry for
    # ``model_name``; pass an explicit ParameterSpace to override.
    param_space: ParameterSpace | None = None

    # Set False when reconstructing inside a task subprocess so we don't
    # truncate/steal the campaign log that the driver owns.
    _fresh_log: bool = True

    run_dir: str = field(init=False)
    data_dir: str = field(init=False)
    log_file: str = field(init=False)
    logger: Any = field(init=False, repr=False, compare=False)


    def __post_init__(self) -> None:
        if self.param_space is None:
            self.param_space = get_parameter_space(self.model_name)
        root = Path.cwd() / _RUNS_ROOT / self.model_name / self.run_label
        self.run_dir = str(root)
        self.data_dir = str(root / "data")
        self.log_file = str(root / "rose.log")
        root.mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        if self._fresh_log:
            # Driver path: open the campaign log fresh, make it the implicit
            # active file (so bare get_logger calls in the driver land here),
            # and record the campaign settings immediately on creation.
            configure_run_log_file(self.log_file, mode="w")
            self.logger = get_logger("rose.run", log_file=self.log_file, mode="w")
            self.log_context()
        else:
            # Reconstruction path: bind a logger that appends, without
            # truncating or hijacking the active file.
            self.logger = get_logger("rose.run", log_file=self.log_file)

    def log_context(self, logger=None) -> None:
        """Write global run settings to the campaign log."""
        log = logger or self.logger
        log_dataclass(log, "GlobalRunContext", self, skip=frozenset({"logger"}))
        log_dataclass(log, "ParameterSpace", self.param_space)

    def create_sub(
        self,
        wf_ID: str = "wf_0",
        *,
        param_space: ParameterSpace | None = None,
        **config_kwargs: Any,
    ) -> SubRunContext:
        """Build a sub-context with global settings and shared data paths injected.

        ``param_space`` is this sub-run's region of concern; ``None`` inherits
        the full campaign space. Narrow it with :meth:`ParameterSpace.narrowed`.
        Any remaining keywords (``al_method``, ``n_select``, …) configure the
        learner via :class:`Config`.
        """
        run_config = Config(wf_ID=wf_ID, **config_kwargs)
        return SubRunContext(
            global_run_context=self,
            run_config=run_config,
            param_space=param_space,
        )

    def print(self) -> None:
        print("GlobalRunContext:")
        for attr, value in self.__dict__.items():
            print(f"    {attr}: {value}")

@dataclass
class Config:
    """Per-sub *learning* settings (what worker scripts read).

    This holds only how a sub-workflow learns — not which region it explores.
    The region lives entirely in :attr:`SubRunContext.param_space`.
    """

    al_method: str = "uncertainty"
    wf_ID: str = "wf_0"
    pod_inc: bool = False
    pod_n_components: int = 20
    n_select: int = 5
    candidate_size: int = 1000
    # Number of training calls between full GP hyperparameter re-optimizations.
    # In-between trainings warm-restart with frozen hyperparameters for speed.
    reoptimize_every: int = 10

    @property
    def wc_ID(self) -> str:
        """Backward-compatible alias for ``wf_ID``."""
        return self.wf_ID


@dataclass
class Artifact:
    """Filesystem paths for one sub-context."""

    run_path: str

    context_file: str = field(init=False)
    log_file: str = field(init=False)
    new_sample_path: str = field(init=False)
    sample_history_path: str = field(init=False)
    surrogate_file: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.run_path)
        self.context_file = str(root / "context.json")
        self.log_file = str(root / "rose.log")
        self.new_sample_path = str(root / "new_sample.pkl")
        self.sample_history_path = str(root / "sample_history.pkl")
        self.surrogate_file = str(root / "surrogate.pkl")

    def as_dict(self) -> dict[str, str]:
        return {
            "context_file": self.context_file,
            "log_file": self.log_file,
            "sample_history_path": self.sample_history_path,
            "new_sample_path": self.new_sample_path,
            "surrogate_file": self.surrogate_file,
        }

    @classmethod
    def from_dict(cls, run_path: str, blob: dict[str, str]) -> Artifact:
        blob = dict(blob)
        if "log_file" not in blob and "logs_path" in blob:
            blob["log_file"] = str(Path(blob["logs_path"]) / "rose.log")
        art = cls(run_path=run_path)
        for key in (
            "context_file",
            "log_file",
            "new_sample_path",
            "sample_history_path",
            "surrogate_file",
        ):
            if key in blob:
                setattr(art, key, blob[key])
        return art


@dataclass
class SubRunContext:
    """One parallel workflow instance under a :class:`GlobalRunContext`."""

    global_run_context: GlobalRunContext
    run_config: Config
    # Region of concern for this sub-run; ``None`` inherits the campaign space.
    param_space: ParameterSpace | None = None

    run_path: str = field(init=False)
    run_artifacts: Artifact = field(init=False)
    logger: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        g = self.global_run_context
        if self.param_space is None:
            self.param_space = g.param_space
        wf_dir = Path(g.run_dir) / self.run_config.wf_ID
        wf_dir.mkdir(parents=True, exist_ok=True)
        self.run_path = str(wf_dir)
        self.run_artifacts = Artifact(run_path=self.run_path)
        # Each sub gets its own log file and its own uniquely named logger
        # (mode="w" => fresh per creation). Bound to an explicit file so it does
        # not perturb the campaign's active file in the driver process.
        self.logger = self._bind_logger(mode="w")
        self.persist()
        self.log_context()

    def _bind_logger(self, *, mode: str = "a"):
        """Bind this sub-run's logger to ``{wf_dir}/rose.log``."""
        return get_logger(
            f"rose.wf.{self.run_config.wf_ID}",
            log_file=self.run_artifacts.log_file,
            mode=mode,
        )

    def log_context(self, logger=None) -> None:
        """Write sub-run settings and artifact paths to this workflow's log."""
        log = logger or self.logger
        log_dataclass(log, "SubRunContext — run_config", self.run_config)
        log_dataclass(log, "SubRunContext — param_space", self.param_space)
        log_dataclass(log, "SubRunContext — artifacts", self.run_artifacts)
        log.info("  run_path: %s", self.run_path)
        log.info("  data_dir: %s", self.global_run_context.data_dir)


    def persist(self) -> str:
        """Write resolved ``context.json`` and return its path."""
        g = self.global_run_context
        payload = {
            "schema_version": _CONTEXT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_path": self.run_path,
            "global": {
                "run_label": g.run_label,
                "model_name": g.model_name,
                "max_iter": g.max_iter,
                "convergence_threshold": g.convergence_threshold,
            },
            "run_config": asdict(self.run_config),
            "parameter_space": self.param_space.to_dict(),
            "artifacts": self.run_artifacts.as_dict(),
        }
        path = self.run_artifacts.context_file
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path


    @classmethod
    def load(cls, context_file: str) -> SubRunContext:
        """Reconstruct from ``context.json`` (v2 snapshot or legacy v1)."""
        with open(context_file) as f:
            blob = json.load(f)
        
        return cls._load_context(blob)
        

    @classmethod
    def _load_context(cls, blob: dict[str, Any]) -> SubRunContext:
        g_blob = blob["global"]
        global_ctx = GlobalRunContext(
            run_label=g_blob["run_label"],
            model_name=g_blob["model_name"],
            max_iter=g_blob["max_iter"],
            convergence_threshold=g_blob["convergence_threshold"],
            _fresh_log=False,
        )
        c_blob = blob["run_config"]
        config_fields = {f.name for f in fields(Config)}
        config = Config(**{k: v for k, v in c_blob.items() if k in config_fields})
        ctx = cls.__new__(cls)
        ctx.global_run_context = global_ctx
        ctx.run_config = config
        ctx.run_path = blob["run_path"]
        ctx.run_artifacts = Artifact.from_dict(
            ctx.run_path, blob.get("artifacts", {})
        )
        # Prefer the persisted resolved space; fall back to the campaign space.
        ps_blob = blob.get("parameter_space")
        ctx.param_space = (
            ParameterSpace.from_dict(ps_blob)
            if ps_blob is not None
            else global_ctx.param_space
        )
        # Task subprocess: pin the sub log as the active file so that bare
        # get_logger(__name__) calls in the task scripts write here, and bind
        # this sub's own logger for direct use.
        configure_run_log_file(ctx.run_artifacts.log_file, mode="a")
        ctx.logger = ctx._bind_logger(mode="a")
        return ctx

    
    def gen_seed(self, seed: np.ndarray) -> str:
        with open(self.run_artifacts.new_sample_path, "wb") as f:
            pickle.dump(seed, f)
        return self.run_artifacts.new_sample_path


    def print(self) -> None:
        print("RunContext:")
        self.global_run_context.print()
        print("SubRunContext:")
        print(f"    run_path: {self.run_path}")
        print("    run_config:")
        for attr, value in asdict(self.run_config).items():
            print(f"      {attr}: {value}")
        print("    param_space:")
        print(f"      name: {self.param_space.name}")
        print(f"      sol_keys: {self.param_space.sol_keys}")
        print(f"      l_bounds: {self.param_space.l_bounds}")
        print(f"      u_bounds: {self.param_space.u_bounds}")
        print("    run_artifacts:")
        for attr, value in self.run_artifacts.as_dict().items():
            print(f"      {attr}: {value}")


if __name__ == "__main__":
    # Campaign space comes from the registry via model_name.
    global_ctx = GlobalRunContext(run_label="tester", model_name="shock_tube")

    # wf_0: inherits the full campaign space.
    sub = global_ctx.create_sub(wf_ID="wf_0", al_method="random", pod_inc=True)
    sub.print()

    # wf_1: narrows to the low-pressure half and a subset of solutions.
    low_p = global_ctx.param_space.narrowed(
        p_high=(75_000.0, 100_000.0), sol_keys=("rho",), name="shock_tube:wf_1"
    )
    sub_2 = global_ctx.create_sub(
        wf_ID="wf_1",
        param_space=low_p,
        al_method="uncertainty",
    )
    sub_2.print()

    # Round-trip through context.json to confirm the resolved space survives.
    reloaded = SubRunContext.load(sub_2.run_artifacts.context_file)
    print("\nReloaded wf_1 param_space:", reloaded.param_space.l_bounds, "->", reloaded.param_space.u_bounds)


    
