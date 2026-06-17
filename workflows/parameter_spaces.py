"""Problem-specific parameter spaces for simulation and surrogate queries.

A :class:`ParameterSpace` is the single source of truth for one problem: the
design bounds active learning samples, the solution fields ("solutions of
interest") the surrogate models, and the snapshot time window. Campaign-wide
spaces live in :data:`DEFAULT_SPACES` and are looked up by model name with
:func:`get_parameter_space`.

Each sub-workflow may *narrow* the campaign space — tightening bounds and/or
picking a subset of solutions of interest — via :meth:`ParameterSpace.narrow`.
The result is validated to be a subset of the parent and is persisted into the
sub-run ``context.json`` so worker scripts can reconstruct it with
:meth:`ParameterSpace.from_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import h5py
import os

from helpers.lhs import lhs

__all__ = [
    "ParameterSpace",
    "DEFAULT_SPACES",
    "get_parameter_space",
]


@dataclass(frozen=True)
class ParameterSpace:
    """Defines bounds, sampling, and surrogate input layout for one problem."""

    name: str
    sim_dim: int
    param_names: tuple[str, ...]
    sol_keys: tuple[str, ...]
    l_bounds: tuple[float, ...]
    u_bounds: tuple[float, ...]
    t_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        if len(self.l_bounds) != self.sim_dim or len(self.u_bounds) != self.sim_dim:
            raise ValueError(f"{self.name}: bounds length must match sim_dim={self.sim_dim}")
        if len(self.param_names) != self.sim_dim:
            raise ValueError(f"{self.name}: param_names length must match sim_dim={self.sim_dim}")
        for i, (lo, hi) in enumerate(zip(self.l_bounds, self.u_bounds)):
            if lo > hi:
                raise ValueError(
                    f"{self.name}: l_bounds[{i}]={lo} must be <= u_bounds[{i}]={hi}"
                )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_random(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        samples = np.column_stack(
            [rng.uniform(lo, hi, size=n) for lo, hi in zip(self.l_bounds, self.u_bounds)]
        )
        return np.atleast_2d(samples)

    def sample_lhs(self, n: int) -> np.ndarray:
        return np.atleast_2d(
            lhs(n, self.sim_dim, list(self.l_bounds), list(self.u_bounds))
        )

    def bounds_array(self) -> np.ndarray:
        """Return design bounds as an ``(sim_dim, 2)`` ``[lo, hi]`` array."""
        return np.column_stack([self.l_bounds, self.u_bounds])

    def construct_X(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        with h5py.File(file_path, "r") as f:
            time = np.asarray(f["t"], dtype=float)
            params = np.array([f.attrs[name] for name in self.param_names], dtype=float)
        
        X = np.column_stack([np.tile(params, (len(time), 1)), time])
        return X
        
    def construct_Y(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        with h5py.File(file_path, "r") as f:
            Y = np.column_stack([np.asarray(f[key], dtype=float) for key in self.sol_keys])

        return Y

    # ------------------------------------------------------------------
    # Per-sub-workflow customization
    # ------------------------------------------------------------------

    def narrow(
        self,
        *,
        l_bounds: tuple[float, ...] | None = None,
        u_bounds: tuple[float, ...] | None = None,
        sol_keys: tuple[str, ...] | None = None,
        name: str | None = None,
    ) -> ParameterSpace:
        """Return a sub-space: tighter bounds and/or a subset of solutions.

        ``None`` arguments inherit from this (parent) space. The new bounds must
        sit inside the parent box and the new ``sol_keys`` must be a subset of
        the parent's, so a narrowed space is always a valid subset of its parent.
        """
        eff_lo = tuple(float(x) for x in l_bounds) if l_bounds is not None else self.l_bounds
        eff_hi = tuple(float(x) for x in u_bounds) if u_bounds is not None else self.u_bounds
        eff_keys = tuple(sol_keys) if sol_keys is not None else self.sol_keys

        if len(eff_lo) != self.sim_dim or len(eff_hi) != self.sim_dim:
            raise ValueError(
                f"{self.name}: narrow() bounds must have length sim_dim={self.sim_dim}"
            )
        for i in range(self.sim_dim):
            if eff_lo[i] < self.l_bounds[i] or eff_hi[i] > self.u_bounds[i]:
                raise ValueError(
                    f"{self.name}: narrowed bounds [{eff_lo[i]}, {eff_hi[i]}] for "
                    f"{self.param_names[i]!r} escape parent "
                    f"[{self.l_bounds[i]}, {self.u_bounds[i]}]"
                )
        unknown = set(eff_keys) - set(self.sol_keys)
        if unknown:
            raise ValueError(
                f"{self.name}: sol_keys {sorted(unknown)} not in parent {self.sol_keys}"
            )

        return replace(
            self,
            name=name or self.name,
            l_bounds=eff_lo,
            u_bounds=eff_hi,
            sol_keys=eff_keys,
        )

    def narrowed(
        self,
        *,
        sol_keys: tuple[str, ...] | None = None,
        name: str | None = None,
        **bounds_by_name: tuple[float, float],
    ) -> ParameterSpace:
        """Narrow by parameter *name* instead of positional bound tuples.

        Each keyword names a design parameter and gives its ``(lo, hi)`` range;
        unmentioned parameters keep the parent's bounds. Equivalent to
        :meth:`narrow` but order-independent and self-documenting::

            space.narrowed(p_high=(75_000.0, 100_000.0), sol_keys=("rho",))
        """
        unknown = set(bounds_by_name) - set(self.param_names)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown param(s) {sorted(unknown)}; "
                f"known: {self.param_names}"
            )
        lo = list(self.l_bounds)
        hi = list(self.u_bounds)
        for pname, (b_lo, b_hi) in bounds_by_name.items():
            i = self.param_names.index(pname)
            lo[i], hi[i] = float(b_lo), float(b_hi)
        return self.narrow(
            l_bounds=tuple(lo), u_bounds=tuple(hi), sol_keys=sol_keys, name=name
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sim_dim": self.sim_dim,
            "param_names": list(self.param_names),
            "sol_keys": list(self.sol_keys),
            "l_bounds": list(self.l_bounds),
            "u_bounds": list(self.u_bounds),
            "t_bounds": list(self.t_bounds),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> ParameterSpace:
        return cls(
            name=str(blob["name"]),
            sim_dim=int(blob["sim_dim"]),
            param_names=tuple(blob["param_names"]),
            sol_keys=tuple(blob["sol_keys"]),
            l_bounds=tuple(float(x) for x in blob["l_bounds"]),
            u_bounds=tuple(float(x) for x in blob["u_bounds"]),
            t_bounds=tuple(float(x) for x in blob["t_bounds"]),
        )

    def print(self) -> None:
        print(f"ParameterSpace: {self.name}")
        print(f"  sim_dim: {self.sim_dim}")
        print(f"  param_names: {self.param_names}")
        print(f"  sol_keys: {self.sol_keys}")
        print(f"  l_bounds: {self.l_bounds}")
        print(f"  u_bounds: {self.u_bounds}")
        print(f"  t_bounds: {self.t_bounds}")


# ---------------------------------------------------------------------------
# Campaign-wide registry (keyed by model_name)
# ---------------------------------------------------------------------------

DEFAULT_SPACES: dict[str, ParameterSpace] = {
    "shock_tube": ParameterSpace(
        name="shock_tube",
        sim_dim=4,
        param_names=("p_high", "p_low", "rho_high", "rho_low"),
        sol_keys=("rho", "momentum", "energy"),
        l_bounds=(75_000.0, 7_500.0, 0.75, 0.115),
        u_bounds=(125_000.0, 12_500.0, 1.25, 0.156),
        t_bounds=(0.0, 0.0006),
    ),
    "cdr": ParameterSpace(
        name="cdr",
        sim_dim=1,
        param_names=("beta",),
        sol_keys=("velocity",),
        l_bounds=(0.0,),
        u_bounds=(10.0,),
        t_bounds=(0.0, 0.25),
    ),
}


def get_parameter_space(name: str) -> ParameterSpace:
    """Look up the campaign-wide parameter space for a model/problem name."""
    try:
        return DEFAULT_SPACES[name]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_SPACES))
        raise ValueError(f"Unknown parameter space {name!r}; known: {known}") from exc
