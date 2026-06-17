# ROSE — active-learning surrogates + Bayesian parameter estimation

A research workspace that builds Gaussian-process surrogates for physics
problems with active learning, and couples them with ensemble / Bayesian
parameter estimation, all orchestrated by the **RADICAL ROSE** library.

There are two distinct things in this tree:

- **[`ROSE/`](ROSE/)** — the upstream RADICAL ROSE library (installable package
  `rose`, v0.3.0). Provides the active-learning orchestration primitives
  (`SequentialActiveLearner`, `ParallelActiveLearner`) on top of
  `radical.asyncflow` + `rhapsody`. Has its own `pyproject.toml` and tests.
- **Everything else at the root** — the *application* that uses `rose` to build
  surrogates for two physics problems (`shock_tube`, `cdr`) and to run parameter
  estimation against them. This is the code normally edited here.

> For deep architectural notes, empirical findings, and gotchas, see
> [`CLAUDE.md`](CLAUDE.md). This README is the short tour.

---

## Research goal

A parallelizable workflow that couples a GPR surrogate with ensemble / Bayesian
parameter estimation, validated against real physics, managed by ROSE.

GPR surrogates are emulators: they replicate states from parameters but are not
grounded in physics, so predictions drift near discontinuities (late-time
ringing / Gibbs artifacts). Data assimilation (EnKF / ES-MDA) and Bayesian
sampling (MCMC / HMC) introduce observations to recover physically-consistent
parameters, and comparing those against the surrogate shows how well it tracks
reality.

Three complementary approaches share the **same inverse problem**: infer the 4
Sod initial-condition parameters `[p_high, p_low, rho_high, rho_low]` from
synthetic density observations at `t = 6e-4`. Truth comes from the exact Sod
solution; the forward model is one of `exact`, `euler`, or `surrogate`.

1. **EnKF / ES-MDA** — ensemble data assimilation (`inference/enkf/`).
2. **MCMC / HMC** — Bayesian posterior sampling (`inference/mcmc/`, `MCMC/`).
3. **GPR surrogate as forward model** — drop the surrogate into any of the above
   to ask how well it lets us recover the true parameters.

---

## Layout

| Path | Role |
|---|---|
| [`ROSE/`](ROSE/) | Upstream `rose` library (orchestration primitives) |
| [`workflows/`](workflows/) | Campaign backbone — `run_context.py`, `parameter_spaces.py` |
| [`task_simulations/`](task_simulations/) | Forward solvers; Sod `sod_exact.py` / `sod_euler.py` |
| [`task_train/`](task_train/) | `Surrogate` (StandardScaler → POD → GPR) |
| [`task_active_learning/`](task_active_learning/) | Sample-selection task |
| [`task_stop_criterion/`](task_stop_criterion/) | MSE stop check |
| [`inference/`](inference/) | EnKF / ES-MDA / MCMC / HMC drivers + `common.py` |
| [`MCMC/`](MCMC/) | Self-contained samplers (`mh_mcmc`, `hmc_mcmc`, `mcmc_common`) |
| [`EnKF/`](EnKF/) | C++ EnKF component (Eigen, header-only) |
| [`dakota_mcmc/`](dakota_mcmc/) | Sandia Dakota MCMC on the Sod inverse problem ([README](dakota_mcmc/README.md)) |
| [`validation/`](validation/) | Surrogate-vs-observations validation (pushforward, PIT) |
| [`training_runs/`](training_runs/) | Campaign outputs (surrogates, contexts) |
| [`training_data/`](training_data/) | HDF5 sim data + validation sets + EnKF ensemble |
| [`analysis/`](analysis/) | Surrogate-uncertainty analysis |

`archive/`, `cgns_reading/`, `rhapsody_sessions/`, `reference_code/` are
scratch/legacy material.

---

## Setup

```bash
source rose_env/bin/activate         # Python 3.13 venv; all commands assume it
pip install ./ROSE                   # installs the `rose` library (editable: -e)
```

---

## Common commands

```bash
# --- ROSE surrogate campaigns ---
python workflow_seq.py               # sequential active learner (one wf_* learner)
python workflow_par.py               # parallel active learners (one wf_* per n_select)

# --- Standalone Sod forward models (importable + CLI) ---
python task_simulations/Shock_Tube/sod_exact.py 1e5 1e4 1.0 0.125 6e-4   # analytic
python task_simulations/Shock_Tube/sod_euler.py 1e5 1e4 1.0 0.125 6e-4   # MUSCL-HLLC

# --- Parameter estimation (all --forward: exact | euler | surrogate) ---
python inference/enkf/es_mda.py --na 4 --forward exact
python inference/mcmc/mcmc_infer.py --forward exact
python inference/mcmc/hmc_infer.py --n-samples 800 --burn 300 --adapt-mass

# --- Dakota MCMC (see dakota_mcmc/README.md) ---
python dakota_mcmc/write_observations.py 0.9e5 0.9e4 0.9 0.1 --seed 0
python dakota_mcmc/run_mcmc.py --obs dakota_mcmc/sod_obs.dat \
  --cells dakota_mcmc/sod_cells.npy --forward euler --plot

# --- Validation: push a chain through the surrogate, check calibration ---
python validation/val.py --chain dakota_mcmc/sod_chain.dat

# --- ROSE library dev (from ./ROSE) ---
cd ROSE && pytest tests/unit
```

---

## Status (high level)

- **ES-MDA** recovers params to ~1% (exact forward), ~1% with the surrogate.
- **MCMC / HMC** give full posteriors; density observations weakly constrain
  pressure (wider `p_high`/`p_low` marginals).
- **Dakota** runs the same inverse problem via DREAM (out of the box) or QUESO
  (after a GSL-enabled rebuild); see [`dakota_mcmc/README.md`](dakota_mcmc/README.md).
- The **run_200** surrogate's failure mode is *time, not parameter space* —
  late-time ringing at the contact/shock; see [`analysis/`](analysis/).

See the TODO and empirical tables in [`CLAUDE.md`](CLAUDE.md) for details.
