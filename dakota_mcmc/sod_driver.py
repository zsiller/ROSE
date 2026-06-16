#!/usr/bin/env python
"""Dakota analysis driver: the Sod forward model as a black box for MCMC.

Dakota's ``bayes_calibration`` method (see ``sod_bayes.in``) calls this script
once per likelihood evaluation via the fork interface::

    sod_driver.py <params_file> <results_file>

We parse the 4 Sod initial-condition parameters Dakota proposed, run the forward
model g(m) -> density, and return ONLY the density at the observed cells (the
``calibration_terms``) so Dakota can form the misfit against ``sod_obs.dat``.

The forward model is selected by the ``SOD_FORWARD`` env var:

    SOD_FORWARD=euler   MUSCL-HLLC solver (default; honest, no inverse crime)
    SOD_FORWARD=exact   analytic Sod      (fast)
    SOD_FORWARD=surrogate  trained GPR

OBSERVED CELLS: the mask is NOT recomputed here -- it depends on the operating
point (cells sit in the Sod region interiors, away from the moving shock/contact)
and so must match exactly the mask used to synthesize the data. ``gen_calibration
_data.py`` writes those indices to ``sod_cells.npy``; we load them, so the model
is evaluated at precisely the cells the data lives at, for any truth, with no
duplicated constants to keep in sync.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# --- make the project + Dakota's Python interface importable ---------------
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_DAKOTA_PY = Path.home() / "dakota-install" / "share" / "dakota" / "Python"
if _DAKOTA_PY.is_dir() and str(_DAKOTA_PY) not in sys.path:
    sys.path.insert(0, str(_DAKOTA_PY))

import dakota.interfacing as di  # noqa: E402

from common import PARAM_NAMES, make_forward  # noqa: E402

# Observed cell indices, written by gen_calibration_data.py -- the single source
# of truth for the observation mask.
_CELLS = Path(__file__).resolve().parent / "sod_cells.npy"


def main() -> None:
    forward = os.environ.get("SOD_FORWARD", "euler")

    params, results = di.read_parameters_file()

    # Pull the 4 ICs in canonical order; works whether Dakota orders them by
    # descriptor or not.
    m = [float(params[name]) for name in PARAM_NAMES]

    if not _CELLS.exists():
        raise FileNotFoundError(
            f"{_CELLS} not found -- run gen_calibration_data.py first to write "
            "the observation mask (and the matching sod_obs.dat).")
    cell_idx = np.load(_CELLS)

    g = make_forward(forward, cell_idx, in_process=True)
    rho_obs = g(m)                             # length == calibration_terms

    if rho_obs.size != results.num_responses:
        raise ValueError(
            f"forward returned {rho_obs.size} values but Dakota expects "
            f"{results.num_responses} calibration_terms -- set "
            f"calibration_terms = {rho_obs.size} in sod_bayes.in")

    for resp, val in zip(results.responses(), rho_obs):
        if resp.asv.function:
            resp.function = float(val)

    results.write()


if __name__ == "__main__":
    main()
