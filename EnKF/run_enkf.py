"""Run one EnKF analysis step on a pre-built shock-tube ensemble.

Unlike ``inference/enkf/enkf_driver.py`` (which forecasts its prior on the fly),
this driver reads an ensemble that ALREADY EXISTS on disk as a set of HDF5
checkpoint files in ``EnKF/ensemble/<set>/`` (or generates one with
``gen_ensemble`` when the directory is empty). Each checkpoint is a sim-format
shock-tube file (rho/momentum/energy datasets + the 4 IC attributes); its final
density snapshot becomes the member's local state and the 4 ICs + time the
augmented globals, giving the 261-row augmented state

    [ p_high, p_low, rho_high, rho_low, t,  rho_0 ... rho_255 ]
     |<------------ 5 globals ----------->|  |<-- 256-cell density -->|

The observations are the EXACT Sod density at the truth operating point
(``task_simulations/Shock_Tube/sod_exact.py``), sampled in the interiors of the
5 Sod regions -- never on a discontinuity. The observation operator H selects
those cells out of the augmented state. The matrix-heavy analysis runs in C++
(``EnKF/enkf_step``) through the bridge in ``EnKF/ensemble_common.py``.

Everything this driver needs lives in the ``EnKF/`` folder. Run it from the repo
root (it self-inserts the root on ``sys.path``).

Typical use
-----------
``--ensemble-dir`` is REQUIRED -- there is no default set. Point it at any
directory of sim-format ``*.h5`` checkpoints (written by ``sod_euler.py --h5``
or ``gen_ensemble``); the driver runs one analysis step and writes the figure to
``EnKF/results/run_enkf.png``::

    python EnKF/run_enkf.py --ensemble-dir EnKF/ensemble/set_1
    python EnKF/run_enkf.py --ensemble-dir EnKF/ensemble/set_2

Build a FRESH ensemble first, then assimilate it. ``--gen N`` forward-solves N
members (ICs drawn as ``truth * (1 + gen_spread * N(0,1))``) into the SAME
``--ensemble-dir`` you pass, so pick a fresh directory unless you mean to add to
an existing one::

    python EnKF/run_enkf.py --gen 20 --ensemble-dir EnKF/ensemble/set_2
    python EnKF/run_enkf.py --gen 40 --gen-spread 0.15 --ensemble-dir EnKF/ensemble/wide

Observation knobs -- how many cells are observed and where. ``--obs-every`` sets
the baseline spacing; ``--margin`` widens the exclusion zone around each
discontinuity (bigger margin => fewer, cleaner interior obs); ``--obs-error`` is
the noise std that sets R::

    python EnKF/run_enkf.py --obs-every 8 --margin 6      # denser obs, wider front guard
    python EnKF/run_enkf.py --obs-error 0.02              # looser obs (weaker pull)

Localization knobs. By default a Gaspari-Cohn taper with a 50-cell cutoff is
applied to the density rows; ``--no-loc`` turns it off, ``--loc-rad`` changes the
cutoff. (Note: localization does NOT taper the 4 global param rows -- the known
global-pull issue.)::

    python EnKF/run_enkf.py --no-loc
    python EnKF/run_enkf.py --loc-rad 25

Reproducibility / output. ``--seed`` drives both the obs perturbation and any
``--gen`` draws; ``--outdir`` redirects the figure; ``--no-plot`` skips it
(handy for a quick RMSE read-out)::

    python EnKF/run_enkf.py --seed 7 --no-plot
    python EnKF/run_enkf.py --outdir /tmp/enkf_run

What it prints: the prior vs analysis density RMSE against the exact Sod field,
and each global parameter's prior mean -> analysis mean next to the truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# This script lives one level under the repo root (EnKF/).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from EnKF.ensemble_common import (  # noqa: E402
    FIELD_OFFSET, N_FIELD, OBS_ERROR, PARAM_NAMES, T_FINAL, TRUTH,
    build_observations, enkf_filter_cpp, gen_ensemble, load_ensemble,
    plot_field_comparison, plot_param_forward_compare, record_parameters,
)

_HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = _HERE / "results"


def run(args) -> dict:
    rng = np.random.default_rng(args.seed)
    localize = not args.no_loc

    # --- prior: HDF5 checkpoints on disk (generate them if asked) -----------
    if args.gen:
        print(f"[gen] forward-solving {args.gen} members into {args.ensemble_dir}")
        ensemble = gen_ensemble(args.gen, args.ensemble_dir, spread=args.gen_spread, rng=rng)
    else:
        ensemble = load_ensemble(args.ensemble_dir)

    # The params the observations are generated from -- i.e. what EnKF tries to
    # recover. The ensemble stays centered on the default; --truth lets you
    # assimilate toward a DIFFERENT operating point than that default.
    truth = TRUTH if args.truth is None else np.asarray(args.truth, dtype=float)

    # --- observations: exact Sod density in the region interiors -----------
    H, obs, cell_idx, exact_rho = build_observations(args.obs_every, args.margin,
                                                     truth=truth)

    # Localization geometry: globals carry a placeholder position (left
    # untapered); local density cells carry their grid index.
    state_loc = np.zeros(ensemble.shape[0])
    state_loc[FIELD_OFFSET:] = np.arange(N_FIELD)
    obs_loc = cell_idx.astype(float)

    print(f"[setup] ensemble_dir={args.ensemble_dir} Ne={ensemble.shape[1]} "
          f"m={cell_idx.size} obs_every={args.obs_every} margin={args.margin} "
          f"(region interiors only) localize={localize} loc_rad={args.loc_rad}")
    print(f"[truth] p_high={truth[0]:.3e} p_low={truth[1]:.3e} "
          f"rho_high={truth[2]:.4f} rho_low={truth[3]:.4f} t={T_FINAL:.2e}")

    # --- one C++ EnKF analysis step ----------------------------------------
    X_a = enkf_filter_cpp(
        ensemble, obs, H, args.obs_error,
        state_loc=state_loc, obs_loc=obs_loc,
        num_globals=FIELD_OFFSET, loc_rad=args.loc_rad, localize=localize,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    prior_mean = ensemble.mean(axis=1)
    post_mean = X_a.mean(axis=1)
    prior_rmse = float(np.sqrt(np.mean((prior_mean[FIELD_OFFSET:] - exact_rho) ** 2)))
    post_rmse = float(np.sqrt(np.mean((post_mean[FIELD_OFFSET:] - exact_rho) ** 2)))

    print(f"[rmse] prior={prior_rmse:.4e}  analysis={post_rmse:.4e}")
    print("[params] inferred global mean (prior -> analysis vs truth):")
    for k, nm in enumerate(PARAM_NAMES):
        print(f"  {nm:9s} {prior_mean[k]:.4e} -> {post_mean[k]:.4e}   "
              f"(truth {truth[k]:.4e})")

    # Record the estimated global parameters (prior/analysis means, stds, the
    # truth, relative error, and the per-member analysis values) to JSON.
    params_path = Path(args.params_out) if args.params_out else \
        Path(args.outdir) / "run_enkf_params.json"
    record_parameters(ensemble, X_a, params_path, truth=truth)

    return {
        "exact": exact_rho,
        "prior_members": ensemble[FIELD_OFFSET:, :],     # (N_FIELD x Ne) prior field
        "post_mean": post_mean[FIELD_OFFSET:],
        "prior_params": ensemble[:FIELD_OFFSET, :],      # (5 x Ne) prior globals
        "post_params": X_a[:FIELD_OFFSET, :],            # (5 x Ne) analysis globals
        "truth": truth,                                  # params the obs came from
        "obs_x": cell_idx / N_FIELD,
        "obs_y": obs,
        "prior_rmse": prior_rmse,
        "post_rmse": post_rmse,
        "params_path": str(params_path),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-dir", dest="ensemble_dir", required=True,
                    help="directory of HDF5 checkpoint files to assimilate (the prior "
                         "ensemble). With --gen, the freshly solved members are written "
                         "here instead; without it, the directory must already hold *.h5")
    ap.add_argument("--truth", type=float, nargs=4, default=None,
                    metavar=("P_HIGH", "P_LOW", "RHO_HIGH", "RHO_LOW"),
                    help="the 4 params the observations are generated from -- the "
                         "operating point EnKF tries to recover (default: the module "
                         "TRUTH = 1e5 1e4 1.0 0.125). The ensemble stays centered on the "
                         "default, so this assimilates toward a different point.")
    ap.add_argument("--gen", type=int, default=0, metavar="N",
                    help="forward-solve N fresh members into --ensemble-dir before assimilating")
    ap.add_argument("--gen-spread", dest="gen_spread", type=float, default=0.1,
                    help="relative IC std used when generating a fresh ensemble (--gen)")
    ap.add_argument("--obs-every", type=int, default=15,
                    help="baseline: observe every k-th density cell in the flat regions")
    ap.add_argument("--margin", type=int, default=4,
                    help="exclude cells within this many of each discontinuity, so "
                         "observations land only in the 5 region interiors")
    ap.add_argument("--obs-error", dest="obs_error", type=float, default=OBS_ERROR,
                    help="observation noise std (R = obs_error^2 I)")
    ap.add_argument("--loc-rad", dest="loc_rad", type=float, default=50.0,
                    help="Gaspari-Cohn localization cutoff (cells)")
    ap.add_argument("--no-loc", action="store_true", help="disable localization")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--params-out", dest="params_out", default=None,
                    help="where to write the estimated-parameter JSON "
                         "(default: <outdir>/run_enkf_params.json)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    res = run(args)
    if not args.no_plot:
        plot_field_comparison(
            exact=res["exact"], members=res["prior_members"],
            post_mean=res["post_mean"], obs_x=res["obs_x"], obs_y=res["obs_y"],
            post_rmse=res["post_rmse"],
            title=f"EnKF analysis on pre-built ensemble  (t = {T_FINAL:.4f})",
            post_label="Analysis mean", prior_label="Prior ensemble",
            save_path=Path(args.outdir) / "run_enkf.png",
        )
        plot_param_forward_compare(
            res["post_params"].mean(axis=1),
            prior_params=res["prior_params"].mean(axis=1),
            truth=res["truth"],
            save_path=Path(args.outdir) / "run_enkf_params.png",
            title="Euler(EnKF-predicted params) vs Euler(truth params)",
        )


if __name__ == "__main__":
    main()
