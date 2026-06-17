---
name: enkf-ensemble-plan
description: How the EnKF ensemble for the shock tube is sourced now vs. the eventual plan
metadata:
  type: project
---

EnKF testing on the shock tube uses an ensemble whose state vector is
`[p_high, p_low, rho_high, rho_low, t_f, rho_0..rho_255]` (length 261), one
member per column — the convention `EnKF::filter` reads. 4 augmented/global ICs
+ static final time + 256-cell density field.

**Now (bootstrap):** `task_simulations/Shock_Tube/gen_enkf_ensemble.py` generates
the ensemble by running real shock-tube sims with log-normally perturbed ICs
(reusing gen_ensemble.py's scheme), plus a sparse density observation operator H
from an unperturbed truth run. Output: `training_data/shock_tube/enkf_ensemble/`.

**Eventual plan:** the ensemble will instead be composed from a ROSE run's
`data_dir` (the campaign's accumulated simulation outputs) with supplemental
members drawn from the trained surrogate. The bootstrap script stands in until
that path exists.

Next steps not yet started: integrating EnKF into the shock-tube loop, and the
cross-validation step comparing the GPR surrogate against the EnKF-inferred state.
See [[[EnKF section of CLAUDE.md]]].
