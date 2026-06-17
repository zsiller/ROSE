# Project Instructions: EnKF/MCMC Parameter Estimation + Surrogate Validation

## What We're Doing

The goal is **parameter estimation and surrogate validation for the 1D shock tube
problem**, using three complementary approaches:

1. **EnKF / ES-MDA** — ensemble-based data assimilation that infers the 4 Sod
   initial-condition parameters `[p_high, p_low, rho_high, rho_low]` from
   synthetic density observations at `t = 6e-4`.
2. **MCMC / HMC** — Bayesian posterior sampling over the same 4 parameters;
   gives full uncertainty quantification (marginals, credible intervals, posterior
   field band) rather than a point estimate.
3. **GPR surrogate validation** — the trained GP surrogate (`run_200`,
   `surrogate.pkl`) replaces the forward model in the above so we can compare
   surrogate-based inference against the exact/Euler gold standard. The surrogate
   maps `[p_high, p_low, rho_high, rho_low, t]` → 256-cell `rho(x)`.

All three approaches share the **same inverse problem**: synthetic "truth"
observations come from the exact Sod solution (`sod_exact.py`), and the forward
model is one of: `exact`, `euler` (numerical, no inverse crime), or `surrogate`
(the GPR).

The point of the comparison is to answer: **how well does the GP surrogate let us
recover the true parameters relative to the physical forward models?** This is the
cross-validation / surrogate-validation step that is listed as TODO in `CLAUDE.md`.

In essence a surrogate has been trained using the RADICAL Orchestrator for Surrogate Exploration (ROSE). This is our workflow engine. It will eventually manage all components of this workflow using parallel and asynchronous execution. More about ROSE and the surrogate isolated workflow can be found in `CLAUDE.md`.

---

## Motivation

Current surrogate models like the GPR currently used are emulators. They learn from training data to replicate states from a set of parameters. The behavior of GPRs is not grounded in physics so they can drift and give nonsense predictions. That is where the various ensemble based and markov chain methods come into play. By introducing some source of observations and estimating parameters we can create a predicted state that is realistic. Then by comparing to the surrogate model we can asses if it follows reality.

---

## Current Challenges

1. Tested on a shock tube which has nonlinear behavior and bimodal distribution around shocks.

2. The non gaussian behavior makes ensemble based approaches less effective because they assume linearity and gaussian behavior.

3. Try and use MCMC based methods, but those are often lengthy and costly.

4. Potential solution use the surrogate to assist in forward propegation and train and optimize while in process.

## Current Script Landscape

These five scripts each solve the same or closely related inverse problem but were
written independently and contain significant duplication:

| Script | Method | Forward | State |
|---|---|---|---|
| `EnKF/EnKF_driver.py` | Single-step augmented-state EnKF (C++ filter) | Euler | 261-row state (4 params + t + 256 density) |
| `EnKF/EnKF_cycle_driver.py` | Time-cycled EnKF (12 cycles) | Euler | same augmented state |
| `EnKF/es_mda.py` | ES-MDA (4–8 iterations, param-space only) | exact or euler | 4 params only |
| `MCMC/mcmc_infer.py` | Random-walk Metropolis | exact or euler | 4 params |
| `MCMC/hmc_infer.py` | Hamiltonian Monte Carlo (finite-diff gradients) | exact or euler | 4 params |

**Known duplication / coupling issues:**

- `EnKF_driver.py` defines the shared constants (`TRUTH`, `T_FINAL`, `N_FIELD`,
  `OBS_ERROR`), the C++ bridge (`enkf_filter_cpp`, `_ensure_enkf_step`),
  observation builder (`build_observations`), exact-Sod wrapper
  (`exact_density_on_cells`), and ensemble loader (`load_ensemble`).
- `es_mda.py`, `mcmc_infer.py`, and `hmc_infer.py` all import from
  `EnKF_driver.py` — there is a fragile `sys.path` insert and a hard dependency on
  the script's location relative to the repo root.
- `hmc_infer.py` additionally imports helpers from `mcmc_infer.py`
  (`make_forward`, `prior_bounds`, `_euler_density`, `make_plots`, `PARAM_NAMES`).
- None of the five scripts has a `--forward surrogate` option yet — adding it
  requires touching each independently.

**Line counts:** `EnKF_driver` 429 / `EnKF_cycle_driver` 376 / `es_mda` 260 /
`mcmc_infer` 399 / `hmc_infer` 383 — 1847 lines total with a lot of repeated
physics setup.

---

## What Needs to Be Done: Streamlining

### Goal

Refactor the five scripts so that:

1. **Shared infrastructure lives in one place**, not spread across driver scripts.
2. **Adding a new forward model** (`surrogate`, or any other) requires touching
   exactly one function.
3. **All five methods** expose a consistent CLI with `--forward exact|euler|surrogate`.
4. Scripts remain individually runnable (`python EnKF/EnKF_driver.py`, etc.) — no
   grand unification that breaks the existing run commands.

### Suggested factoring

Extract a shared module — something like `inference_common.py` (or
`EnKF/inference_common.py`) — containing:

- **Constants:** `TRUTH`, `T_FINAL`, `N_FIELD`, `N_PARAMS`, `OBS_ERROR`, `GAMMA`,
  `PARAM_NAMES`, `PRIOR_MEAN`, `PRIOR_SPREAD`.
- **Forward models:** `make_forward(mode: str, cell_idx) -> Callable` dispatching
  on `exact | euler | surrogate`. The surrogate branch loads
  `training_runs/shock_tube/run_200/wf_0/surrogate.pkl` (with the module alias fix
  for unpickling — see surrogate-pickle-module-alias memory).
- **Observation builder:** `build_observations(args)` — currently duplicated
  between `EnKF_driver` and `es_mda`.
- **C++ bridge:** `enkf_filter_cpp`, `_ensure_enkf_step` — used only by the two
  EnKF drivers but should live alongside the rest of the shared infrastructure.
- **Plotting utilities:** basic comparison plot (exact / prior ensemble / posterior
  mean) reusable by all methods.

Each of the five scripts then becomes a thin wrapper: parse args, call
`make_forward`, run method, plot.

### Surrogate forward model detail

The GPR surrogate (`run_200/wf_0/surrogate.pkl`) predicts `rho(x)` at 256 cells
given `[p_high, p_low, rho_high, rho_low, t]`. When used as a forward model:

- Load once at startup (expensive — don't reload per call).
- Unpickle with the module-alias patch:
  ```python
  import sys, types, task_train.model as _m
  sys.modules.setdefault("surrogate", types.ModuleType("surrogate"))
  sys.modules["surrogate"].Surrogate = _m.Surrogate
  ```
- Call `surrogate.predict(X)` where `X` has shape `(n, 5)`.
- The surrogate's output is in physical units (inverse-transformed by the scaler);
  index out only the `rho` field using `sol_keys`.

### Concrete deliverable

A refactored directory where:

```
inference_common.py        # shared constants + forward factories + obs builder
EnKF/
  EnKF_driver.py           # thin: imports inference_common, runs single-step EnKF
  EnKF_cycle_driver.py     # thin: imports inference_common, runs cycled EnKF
  es_mda.py                # thin: imports inference_common, runs ES-MDA
MCMC/
  mcmc_infer.py            # thin: imports inference_common, runs RW-MCMC
  hmc_infer.py             # thin: imports inference_common, runs HMC
```

With the `--forward surrogate` path added to `es_mda.py`, `mcmc_infer.py`, and
`hmc_infer.py` (those are the parameter-estimation methods that will benefit most).

---

## Key Results to Preserve (Do Not Regress)

These benchmarks should still pass after refactoring (run with `--forward exact`):

| Method | Metric | Target |
|---|---|---|
| EnKF single-step | density RMSE | ~7.9e-3 |
| ES-MDA (Na=4) | density RMSE | ~9.8e-4; params <2% error |
| ES-MDA (euler forward) | param recovery | <2.5% error |
| MCMC (20k steps) | rho_high posterior mean | 0.9998±4e-3; p std ~3% |
| HMC (800 samples, adapt-mass) | acceptance | ~0.91; ESS ~260–320 |

---

## Files to Know

| Path | Role |
|---|---|
| `sod_exact.py` | Analytic Sod solution — truth source |
| `sod_euler.py` | MUSCL-HLLC solver — no-inverse-crime forward model |
| `EnKF/EnKF_driver.py` | Single-step EnKF + shared infrastructure (currently) |
| `EnKF/es_mda.py` | ES-MDA parameter estimation |
| `MCMC/mcmc_infer.py` | RW-Metropolis |
| `MCMC/hmc_infer.py` | HMC (imports from mcmc_infer) |
| `training_runs/shock_tube/run_200/wf_0/surrogate.pkl` | GPR surrogate to add as forward model |
| `task_train/model.py` | `Surrogate` class definition (needed for unpickling) |
| `workflows/parameter_spaces.py` | `ParameterSpace`, `sol_keys`, `construct_Y` |
| `analysis/surrogate_uncertainty.py` | Existing surrogate uncertainty analysis |

---

## Environment

```bash
source rose_env/bin/activate   # Python 3.13 venv
# Run any script from the repo root, e.g.:
python EnKF/EnKF_driver.py
python EnKF/es_mda.py --na 4 --forward euler
python MCMC/mcmc_infer.py --forward exact
python MCMC/hmc_infer.py --n-samples 800 --burn 300 --adapt-mass
```

The C++ `enkf_step` binary is auto-compiled on first use by `_ensure_enkf_step()`
in `EnKF_driver.py` — no manual build step needed for the EnKF methods.
