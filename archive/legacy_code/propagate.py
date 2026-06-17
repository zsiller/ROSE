"""Time propagation of one shock-tube ensemble member for the cycled EnKF.

A thin wrapper over :class:`EulerSolver1D` that lets a cycle driver:

  1. initialise a member from its IC parameters (Sod step at t = 0);
  2. *input the current density state and parameters* after each analysis
     (``reinject``) -- the density is overwritten by the assimilated field while
     the forecast velocity & pressure are kept, and the inferred parameters are
     stored alongside; and
  3. forecast the member forward to the next observation time (``forecast``).

State the cycle hands in/out per member:
  - parameters : ``[p_high, p_low, rho_high, rho_low]``  (augmented globals)
  - density    : length-``nx`` field on the cell grid     (assimilated local state)

This is the propagation half of the cycled EnKF; the analysis half runs in the
C++ root filter (EnKF/enkf_step). Velocity & pressure live in the solver between
cycles, matching the density-only re-injection used across the shock-tube work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# This module lives two levels under the repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task_simulations.Shock_Tube.sod_euler import EulerSolver1D


class ShockTubeMember:
    """One ensemble member: an Euler solver plus its inferred IC parameters."""

    def __init__(self, params, *, nx: int = 256, x0: float = 0.5,
                 gamma: float = 1.4, cfl: float = 0.5):
        self.params = np.asarray(params, dtype=float).copy()  # [p_hi, p_lo, rho_hi, rho_lo]
        self.x0 = x0
        self.solver = EulerSolver1D(nx=nx, xmin=0.0, xmax=1.0, gamma=gamma, cfl=cfl)
        # Initial condition: Sod-like step from the member's parameters at t = 0.
        self.solver.set_sod_like(rho_high=self.params[2], p_high=self.params[0],
                                 rho_low=self.params[3], p_low=self.params[1], x0=x0)

    # -- accessors --------------------------------------------------------
    @property
    def t(self) -> float:
        return self.solver.t

    @property
    def density(self) -> np.ndarray:
        return self.solver.U[0].copy()

    def primitive(self) -> np.ndarray:
        """Current primitive state [rho, u, p]."""
        return self.solver.primitive()

    # -- inputs from the analysis ----------------------------------------
    def reinject(self, density, params=None) -> None:
        """Input the assimilated density state (and optionally the inferred
        parameters) for the next propagation.

        The forecast velocity & pressure are retained; only density is replaced
        (clamped positive). ``params`` updates the member's tracked ICs.
        """
        W = self.solver.primitive()                         # forecast rho, u, p
        rho = np.maximum(np.asarray(density, dtype=float), 1e-6)  # keep positivity
        self.solver.set_state_primitive(rho, W[1], W[2])
        if params is not None:
            self.params = np.asarray(params, dtype=float).copy()

    def set_state(self, rho, u, p) -> None:
        """Overwrite the full primitive state (rho, u, p)."""
        self.solver.set_state_primitive(np.asarray(rho, dtype=float),
                                        np.asarray(u, dtype=float),
                                        np.asarray(p, dtype=float))

    def reproject(self, params, t_target: float) -> int:
        """Reproject the member onto the model's solution manifold.

        Rebuilds the member as the NUMERICAL Sod solution for ``params`` (Sod
        step IC, marched to ``t_target``) and replaces its full (rho, u, p)
        state with that consistent, sharp profile. Unlike ``reinject`` (which
        takes a raw, possibly overshooting analysis density), this guarantees a
        physically valid profile, so the EnKF linear-update overshoot at the
        discontinuities is removed by construction and the state is exactly the
        forward model of the (analyzed) parameters. Params are clamped positive
        so a stray negative analysis can't break the solver. Returns substeps.
        """
        self.params = np.maximum(np.asarray(params, dtype=float), 1e-6)
        self.solver.set_sod_like(rho_high=self.params[2], p_high=self.params[0],
                                 rho_low=self.params[3], p_low=self.params[1], x0=self.x0)
        return self.solver.step_to(t_target)

    # -- forecast ---------------------------------------------------------
    def forecast(self, t_target: float) -> int:
        """Propagate the member to ``t_target`` with CFL-controlled substeps.

        Returns the number of substeps taken (0 if already at/after t_target).
        """
        return self.solver.step_to(t_target)
