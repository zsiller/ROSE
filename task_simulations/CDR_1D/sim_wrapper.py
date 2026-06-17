from __future__ import annotations

import subprocess
import sys
import pandas as pd
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from helpers.h5_read import read_cdr_1d_h5

def run_cdr(data_path: str, *params: float, t_final: float = 0.25, write_every: int = 1000):

    project_root = Path(__file__).resolve().parents[2]

    sim_dir = project_root / "simulations" / "CDR_1D"
    exe = sim_dir / "hdf5_cdr"

    if not exe.is_file():
        raise FileNotFoundError(exe)

    # Keep the simulator quiet so callers (e.g. ROSE stop metrics) can own stdout.
    subprocess.run(
        [str(exe), str(data_path), str(t_final), str(params[0]), str(write_every)],
        cwd=sim_dir,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    return data_path