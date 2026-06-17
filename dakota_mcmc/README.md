# Dakota MCMC for the Sod shock-tube inverse problem

Bayesian calibration of the 4 Sod initial-condition parameters

    m = [p_high, p_low, rho_high, rho_low]

from final-time (`t = 6e-4`) density observations, driven by **Sandia Dakota**.
Dakota's `make_plots` lives in `common.py`; the problem constants, observation
builder, forward models and ensemble loader are imported from the global
`helpers/inverse_common.py` (the single source of truth shared with `EnKF/` and
`MCMC/`). `plots.py` holds surrogate-validation figures (PIT, violins).

## Pieces

| File | Role |
|---|---|
| `common.py` | Dakota `make_plots`; re-exports the shared inverse-problem layer |
| `../helpers/inverse_common.py` | Global constants, `build_observations`, forward models, `load_ensemble` (shared with `EnKF/` + `MCMC/`) |
| `plots.py` | Surrogate pushforward / PIT validation figures |
| `write_observations.py` | Write `sod_obs.dat` + `sod_cells.npy` for a param set |
| `run_mcmc.py` | Stage inputs, run Dakota, optional `--plot` |
| `gen_proposal_cov.py` | Write `prop_cov.dat` from the HDF5 ensemble |
| `sod_driver.py` | Analysis driver: params → Sod forward → observed densities |
| `sod_driver.sh` | Fork wrapper under `rose_env/bin/python` |
| `sod_bayes.in` | Generated Dakota input (via `run_mcmc.py`) |

## The black box

The forward model Dakota calls each likelihood evaluation is the project's
standalone Sod solver, selected by the `SOD_FORWARD` env var:

* `exact` (default) — analytic Sod, fast.
* `euler` — MUSCL-HLLC numerical solver; **honest, no inverse crime** (the truth
  observations come from `exact`, so calibrating with `euler` avoids fitting a
  model to data it generated).
* `surrogate` — trained GPR.

## Run

```bash
source ../rose_env/bin/activate

# 1. observations (exact + Gaussian noise at OBS_ERROR)
python dakota_mcmc/write_observations.py 0.9e5 0.9e4 0.9 0.1 --seed 0

# 2. MCMC (+ optional plots)
python dakota_mcmc/run_mcmc.py \
  --obs dakota_mcmc/sod_obs.dat \
  --cells dakota_mcmc/sod_cells.npy \
  --forward euler \
  --plot --plot-truth 0.9e5 0.9e4 0.9 0.1 --plot-burn 20
```

## Outputs

* `sod_bayes.out` — run log; ends with **"Sample moment statistics for each
  posterior variable"** (posterior mean / std / skew / kurtosis per param).
* `sod_chain.dat` — exported MCMC chain (params + responses per sample).
* `sod_tabular.dat` — all evaluations (tabular).
* `dakota_dream_chain*.txt`, `dakota_dream_gr.txt` — DREAM per-chain samples and
  Gelman–Rubin convergence diagnostic.

## Notes

* **DREAM, not QUESO.** This Dakota build (6.24) was compiled without GSL, so
  `bayes_calibration queso`/`gpmsa` are unavailable. DREAM is Dakota's native
  differential-evolution MCMC and needs no external library. Swapping back to
  `queso dram` only requires a GSL-enabled rebuild.
* The observation mask (`OBS_EVERY=15`, `MARGIN=4` → 16 interior cells) is built
  once by `write_observations.py` (via the shared `build_observations`) and
  written to `sod_obs.dat` + `sod_cells.npy`; `sod_driver.py` only *loads*
  `sod_cells.npy`, so there are no duplicated constants to keep in sync.
* `variance_type 'scalar'` in `sod_bayes.in` means the trailing 16 columns of
  `sod_obs.dat` are **variances** (σ² = `OBS_ERROR`² = 1e-4), one per cell.
* Pressures (`p_high`, `p_low`) are weakly constrained by density observations —
  expect wider posteriors there, consistent with the in-repo samplers.
