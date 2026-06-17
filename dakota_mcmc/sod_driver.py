#!/usr/bin/env python
"""Dakota analysis driver: the Sod forward model as a black box for MCMC.

Dakota calls this once per likelihood evaluation through its fork interface:

    sod_driver.py <params_file> <results_file>

It reads the 4 proposed Sod ICs, runs the forward model, and writes the density
at the observed cells so Dakota can score it against ``sod_obs.dat``. Two env
vars (exported by ``run_mcmc.py``) configure it; both default so it also runs
standalone:

    SOD_FORWARD   exact | euler | surrogate          (default: euler)
    SOD_CELLS     observation-cell .npy to evaluate   (default: ./sod_cells.npy)

The cell mask is loaded, not recomputed, so it always matches the mask
``write_observations.py`` used to build ``sod_obs.dat``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                         # so `import common` works

# Dakota's bundled Python package provides the fork-interface helpers below.
_DAKOTA_PY = Path.home() / "dakota-install" / "share" / "dakota" / "Python"
if _DAKOTA_PY.is_dir():
    sys.path.insert(0, str(_DAKOTA_PY))

import dakota.interfacing as di  # noqa: E402

from common import PARAM_NAMES, make_forward  # noqa: E402


def main() -> None:
    forward = os.environ.get("SOD_FORWARD", "euler")
    cells = Path(os.environ.get("SOD_CELLS", HERE / "sod_cells.npy"))
    if not cells.is_file():
        raise FileNotFoundError(
            f"{cells} not found -- run write_observations.py first to create the "
            "observation mask (and the matching sod_obs.dat).")

    params, results = di.read_parameters_file()
    ics = [float(params[name]) for name in PARAM_NAMES]   # the 4 Sod ICs Dakota proposed

    g = make_forward(forward, np.load(cells), in_process=True)
    rho_at_cells = g(ics)

    if rho_at_cells.size != results.num_responses:
        raise ValueError(
            f"forward produced {rho_at_cells.size} values but Dakota expects "
            f"{results.num_responses} calibration_terms -- set calibration_terms "
            f"= {rho_at_cells.size} in sod_bayes.in")

    for response, value in zip(results.responses(), rho_at_cells):
        if response.asv.function:
            response.function = float(value)
    results.write()


if __name__ == "__main__":
    main()
