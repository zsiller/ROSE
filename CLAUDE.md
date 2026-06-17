# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this repo is

A **research workspace** that drives active-learning surrogate-building campaigns on top of the
`rose` library. There are two distinct things here:

- **`./ROSE/`** — the upstream RADICAL ROSE library (installable package `rose`, version 0.3.0).
  It provides the active-learning orchestration primitives (`SequentialActiveLearner`,
  `ParallelActiveLearner`) built on `radical.asyncflow` + `rhapsody` execution backends.
  This subtree has its own `pyproject.toml`, tests, and tooling.
- **Everything else at the root** — the *application* that uses `rose` to build surrogates for
  two physics problems (`shock_tube`, `cdr`). This is the code you normally edit.

The installed `rose` package points at `./ROSE` (`pip install ./ROSE`), so editing the library
requires a reinstall unless installed editable.

---

## Research Goal

The long-term goal is a **parallelizable workflow** that couples a GPR surrogate with ensemble /
Bayesian parameter estimation, validated against real physics, and managed by ROSE.

Current surrogate models like the GPR are emulators: they learn from training data to replicate
states from a set of parameters. GPR behavior is not grounded in physics, so predictions can drift
or become nonsensical (especially late-time ringing / Gibbs artifacts near discontinuities). That is
where EnKF, ES-MDA, and MCMC come in: by introducing observations and estimating parameters we
produce a predicted state that is physically consistent. Comparing that against the surrogate reveals
how well it tracks reality.

The three complementary approaches all share the **same inverse problem**: infer the 4 Sod
initial-condition parameters `[p_high, p_low, rho_high, rho_low]` from synthetic density observations
at `t = 6e-4`. Truth observations come from the exact Sod solution
(`task_simulations/Shock_Tube/sod_exact.py`); the forward model is one of `exact`, `euler`, or
`surrogate` (dispatched by `inference/common.py::make_forward`).

1. **EnKF / ES-MDA** — ensemble-based data assimilation. ES-MDA (param-space only, 4 iterations)
   beats single-step augmented-state EnKF ~7x on density RMSE and recovers params to <2% error.
2. **MCMC / HMC** — Bayesian posterior sampling; gives full uncertainty quantification (marginals,
   credible intervals, posterior field band) rather than a point estimate.
3. **GPR surrogate as forward model** — replace `exact`/`euler` with the trained surrogate in all of
   the above to answer: *how well does the surrogate let us recover true parameters?* This is the
   cross-validation step listed as TODO below and is the **primary next deliverable**.

### Known physics challenge

The shock tube has non-linear behavior and bimodal distributions around discontinuities (contact
surface + shock). EnKF and ES-MDA assume Gaussian/linear behavior, so they hit a ceiling. MCMC/HMC
are exact but expensive. The planned mitigation is to use the surrogate to assist forward propagation
and to train/optimize it in-process (surrogate-assisted MCMC).

---

## Current status of EnKF / MCMC work

### Empirical findings

| Method | Key result |
|---|---|
| EnKF single-step (C++) | RMSE 1.87e-2 → 7.9e-3; prior-limited, localization doesn't help globals |
| EnKF cycled (12 cycles) | Holds analysis ~1.0e-2; density-only inflation 1.3 is sweet spot |
| ES-MDA (Na=4, exact) | RMSE 9.76e-4; params <2% error; no Gibbs overshoot |
| ES-MDA (euler, honest) | Params <2.5%; RMSE vs exact 1.03e-2 (Euler smear, not ES-MDA failure) |
| MCMC RW-Metropolis | rho_high 0.9998±4e-3, pressures ~3% std (density obs barely see pressure) |
| HMC (800 samples) | Acceptance ~0.91, ESS ~260–320 (0.32–0.40 per sample) |

### Global-pull issue (augmented-state EnKF)

Augmented-state EnKF injects spurious pull on global params (e.g. p_high 1e5 → ~9.1e4) because
distance-based localization can't taper global rows. ES-MDA's param-space iteration is the correct
fix — confirmed by the 7x improvement. For the EnKF cycled driver this is a known open issue.

---

## Inference layout (`inference/`)

All five inference scripts are thin wrappers around `inference/common.py`, which owns the shared
infrastructure (the 2026-06 streamlining refactor). Adding a forward model means touching exactly
one function: `common.make_forward`.

```
inference/
  common.py                   # constants + forward-model factory + obs builder + prior helpers
                              # (load_ensemble, forecast_ensemble, prior_bounds) + C++ bridge
                              # (enkf_filter_cpp) + shared comparison figure
  enkf/
    enkf_driver.py            # single-step augmented-state EnKF (C++ filter), on-the-fly forecast
    enkf_cycle_driver.py      # time-cycled EnKF (uses task_simulations/Shock_Tube/propagate.py)
    es_mda.py                 # ES-MDA, param-space only
    results/                  # figures from the three drivers
  mcmc/
    mcmc_infer.py             # RW-Metropolis (Haario adaptive-cov)
    hmc_infer.py              # HMC, finite-diff gradients (imports make_plots from mcmc_infer)
    results/  figures/        # sampler outputs + LaTeX write-ups
```

All scripts run from the repo root (`python inference/enkf/es_mda.py …`) and self-insert the repo
root on `sys.path`. Every script that takes `--forward` accepts `exact | euler | surrogate`.
`EnKF/` (capitalized) now holds ONLY the C++ component (headers, `enkf_step.cpp`, the
`build_ensemble` demo and its `plot_enkf.py`).

### Surrogate forward model

`common.surrogate_density` / `make_forward("surrogate")` use the GPR surrogate
(`training_runs/shock_tube/run_200/wf_0/surrogate.pkl` by default, lazily loaded + cached): maps
`[p_high, p_low, rho_high, rho_low, t]` → 256-cell `rho(x)` (first N_FIELD columns of `predict`'s
output; sol_keys order rho, momentum, energy). The `surrogate.* → task_train.*` module aliases
needed to unpickle older campaign pickles are applied inside `common.load_surrogate`.

### Reference numbers (current obs methodology, default seeds/flags)

The interiors-only observation mask (`obs_every=15`, `margin=4`, m=16) replaced the older
jump-bracket mask, so older absolute RMSE benchmarks (e.g. ES-MDA 9.8e-4) are not comparable.
Current references:

| Method | Result |
|---|---|
| EnKF single-step (euler forecast prior) | prior RMSE 1.34e-2 → analysis 1.18e-2 |
| ES-MDA (Na=4, exact) | RMSE 5.88e-3; params <1% |
| ES-MDA (Na=4, surrogate) | params ~1%; RMSE vs g(truth) 2.46e-3 |
| MCMC (exact) | rho_high 0.9998±4.4e-3 (matches pre-refactor benchmark) |
| HMC | acceptance ~0.9 |

---

## TODO

- [x] Extract shared infrastructure (`inference/common.py`) and thin out the five inference scripts
- [x] Add `--forward surrogate` to `es_mda.py`, `mcmc_infer.py`, `hmc_infer.py` (and `enkf_driver.py`)
- [ ] Run surrogate-forward comparison: does the GPR recover params as well as `euler`?
      (first datapoint: ES-MDA surrogate recovers params to ~1% — comparable to euler's <2.5%;
      still to do for MCMC/HMC)
- [ ] Integrate EnKF / ES-MDA into the ROSE campaign loop (`RunContext` accepts surrogate class)
- [ ] Cross-validation: compare surrogate prediction vs EnKF-inferred state
- [ ] Test with parallel workflow (`workflow_par.py`)
- [ ] Widen `DEFAULT_KERNEL` length-scale bounds (currently hitting 1e3 upper bound)

---

## Environment & commands

The venv is `rose_env/` (activate with `source rose_env/bin/activate`). All commands below assume it
is active.

```bash
# ROSE surrogate campaigns
python workflow_seq.py        # sequential active learner, one wf_* learner
python workflow_par.py        # parallel active learners, one wf_* per n_select

# Run a single pipeline task standalone
python task_simulations/sim.py        --ctx training_runs/shock_tube/<run>/wf_0/context.json
python task_train/train.py            --ctx <.../context.json>
python task_active_learning/active_learning.py --ctx <.../context.json>
python task_stop_criterion/check_mse.py        --ctx <.../context.json>

# Parameter estimation / data assimilation (all --forward: exact | euler | surrogate)
python inference/enkf/enkf_driver.py                       # single-step EnKF
python inference/enkf/enkf_cycle_driver.py --cycles 12     # cycled EnKF
python inference/enkf/es_mda.py --na 4 --forward exact     # ES-MDA
python inference/mcmc/mcmc_infer.py --forward exact        # RW-Metropolis
python inference/mcmc/hmc_infer.py --n-samples 800 --burn 300 --adapt-mass  # HMC

# Standalone Sod forward models (importable + CLI)
python task_simulations/Shock_Tube/sod_exact.py 1e5 1e4 1.0 0.125 6e-4   # analytic Sod
python task_simulations/Shock_Tube/sod_euler.py 1e5 1e4 1.0 0.125 6e-4   # MUSCL-HLLC numerical

# Cycled EnKF (older Python-only driver in task_simulations/)
python task_simulations/Shock_Tube/cycle_enkf.py --N 20 --frequency 10

# C++ EnKF demo (build first)
h5c++ -std=c++17 -I/usr/include/eigen3 EnKF/build_ensemble.cpp -o EnKF/build_ensemble
./EnKF/build_ensemble training_data/shock_tube/enkf_ensemble_files
python EnKF/plot_enkf.py

# ROSE library dev (from ./ROSE)
cd ROSE && pytest tests/unit
tox -e lint    # ruff check + format --check
tox -e format  # ruff format + check --fix
```

Note: `rose_env` is Python 3.13; ROSE library declares support for 3.10–3.12.

The C++ `enkf_step` binary is auto-compiled on first use by `_ensure_enkf_step()` — no manual build
step needed.

---

## Core architecture (ROSE campaign)

### The campaign → sub → task model

`workflows/run_context.py` is the backbone. Read its module docstring first.

- **`GlobalRunContext`** = one campaign. Owns `training_runs/{model_name}/{run_label}/`, the shared
  `data/` dir, the campaign `rose.log`, and the campaign-wide `ParameterSpace`.
  `.create_sub(wf_ID=..., param_space=..., **config)` spawns a `SubRunContext`.
- **`SubRunContext`** = one `wf_*` learner. Owns `wf_*/` with its own `rose.log`, `context.json`,
  `new_sample.pkl`, `sample_history.pkl`, `surrogate.pkl`. Holds a `Config` (learning settings:
  `al_method`, `n_select`, `pod_inc`, …) and its own (optionally narrowed) `ParameterSpace`.
- **`Artifact`** derives all per-sub file paths from the `wf_*` dir.

### context.json handoff

Drivers do **not** call Python functions directly. Each decorated task returns a **shell command
string** run as a subprocess, e.g. `python .../task_train/train.py --ctx <file>`. The only thing
passed is the path to `context.json`; tasks reconstruct everything via `SubRunContext.load(args.ctx)`.
Follow this pattern for any new task: parse `--ctx`, call `SubRunContext.load`, pull
`g, c, a = ctx.global_run_context, ctx.run_config, ctx.run_artifacts`.

### ParameterSpace

`workflows/parameter_spaces.py` defines `ParameterSpace` (frozen dataclass: design bounds,
`param_names`, `sol_keys`, `t_bounds`). Registry `DEFAULT_SPACES`, looked up via
`get_parameter_space(model_name)`. Owns `sample_lhs`, `construct_X`, `construct_Y` (HDF5 → surrogate
inputs/targets).

### Surrogate pipeline

`task_train/model.py::Surrogate` = `StandardScaler` on X and Y → POD (`task_train/POD.py`, a PCA
wrapper) → `GaussianProcessRegressor` (Matern + WhiteKernel) mapping scaled X to POD coefficients.
`predict` inverse-transforms to physical units. Warm-restarts; re-optimizes GP hyperparameters every
`REOPTIMIZE_EVERY` (=10) trainings. MSE in `check_mse.py` is in *scaled* space.

Known issue: `DEFAULT_KERNEL` length-scale bounds `(1e-3, 1e3)` are too tight — scales hit the 1e3
upper bound (ConvergenceWarning).

### Simulators

`task_simulations/sim.py` dispatches via `SIM_RUNNERS` to:
- `shock_tube` — pure-Python 1D Euler solver (`run_shock_tube.py`, built on
  `Shock_Tube/sod_euler.py::EulerSolver1D`), writes HDF5.
- `cdr` — compiled C++ binary (`CDR_1D/`, invoked via `subprocess`).

The two Sod solvers live in `task_simulations/Shock_Tube/` and are BOTH importable modules and
standalone CLIs (they replaced the old `euler1d.py` / `exact_sod.py` in 2026-06):
- `sod_euler.py` — MUSCL-HLLC `EulerSolver1D` (step_to, run_with_snapshots, HDF5 save/restart,
  `set_state_primitive` for assimilation re-injection).
- `sod_exact.py` — analytic Sod (`exact_state`, plus the array-of-params conveniences
  `exact_density_on_cells` / `shock_features` the inference code calls).

### EnKF (`EnKF/`) — C++ data assimilation

Header-only C++ built on **Eigen**. Two pieces:

- **`ensembleParams` (`Ensemble_Family.h`)** — builds the ensemble. State is
  `(num_globals + n_local) × ens_size`: global params stacked on local state. Globals perturbed with
  Gaussian noise rescaled so sample std exactly equals `pert_g`; locals replicated with no noise.
- **`EnKF` (`EnKF.h`)** — one stochastic perturbed-observation analysis step: background covariance
  `P = cov(X^f)`; innovation `S = HᵀPH + R`; gain `K = PH S⁻¹`; update
  `X^a = X^f + K(yᵒ_e − HᵀX^f)`. Gaspari–Cohn Schur-product localization wired but does not fix
  global-param pull. Pseudo-inverse solve (placeholder).

Known issue: instability / ensemble collapse as number of observations grows.

Gotcha: `#include "../src/Ensemble_Family.h"` paths are stale — headers live in `EnKF/`, so adjust
`-I` when compiling.

### Logging

`helpers/log.py` keeps a registry of file handlers keyed by resolved path for concurrent campaign +
sub logs. Use `GlobalRunContext.logger` for campaign events, `SubRunContext.logger` for one learner.
Inside task subprocesses `SubRunContext.load` sets `ROSE_LOG_FILE`, so a bare
`get_logger(__name__)` auto-routes — don't pass an explicit file there.

---

## Layout notes & gotchas

- Task scripts rely on `workflows/path_setup.ensure_project_root` + `sys.path` insert to run as
  both modules and standalone scripts.
- `main.py` is dead/legacy — imports modules that no longer exist.
- `archive/`, `reference_code/`, `cgns_reading/`, `rhapsody_sessions/` are scratch/legacy material.
- Run outputs land in `training_runs/`; validation data lives in
  `training_data/{model}/validation_set/`.
- `run_200` surrogate (`training_runs/shock_tube/run_200/wf_0/surrogate.pkl`) was pickled under the
  old `surrogate.*` module name. To load it apply module aliases (see Surrogate forward model section
  above). `run_50` surrogates load without this; `setdefault` makes the block harmless there.
- `run_200` GPR failure mode is **time, not param space**: RMSE vs Euler grows ~2e-4 at t=6e-5 to
  ~5e-3 at t=6e-4 (ringing / Gibbs at contact + shock). GP predictive std is a comb (collapses at
  trained time-slices, spikes between) and weakly predicts true error (Pearson r~0.21).

---

## Key file index

| Path | Role |
|---|---|
| `task_simulations/Shock_Tube/sod_exact.py` | Analytic Sod solution — truth / observation source |
| `task_simulations/Shock_Tube/sod_euler.py` | MUSCL-HLLC finite-volume solver — no-inverse-crime forward model |
| `inference/common.py` | ALL shared inference infrastructure (constants, make_forward, obs builder, C++ bridge) |
| `inference/enkf/enkf_driver.py` | Single-step EnKF |
| `inference/enkf/enkf_cycle_driver.py` | Time-cycled EnKF (12 cycles, density-only obs) |
| `inference/enkf/es_mda.py` | ES-MDA parameter estimation |
| `inference/mcmc/mcmc_infer.py` | RW-Metropolis posterior sampler |
| `inference/mcmc/hmc_infer.py` | HMC posterior sampler (imports make_plots from mcmc_infer) |
| `task_simulations/Shock_Tube/cycle_enkf.py` | Older Python-only cycled EnKF driver |
| `task_simulations/Shock_Tube/propagate.py` | `ShockTubeMember` — EulerSolver1D wrapper for cycled driver |
| `training_runs/shock_tube/run_200/wf_0/surrogate.pkl` | GPR surrogate behind `--forward surrogate` |
| `task_train/model.py` | `Surrogate` class (needed for unpickling run_200) |
| `workflows/parameter_spaces.py` | `ParameterSpace`, `sol_keys`, `construct_Y` |
| `analysis/surrogate_uncertainty.py` | Surrogate uncertainty analysis for run_200 |
