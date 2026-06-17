"""Parameter-inference package for the shock-tube inverse problem.

All methods share the same setup (see ``inference/common.py``): infer the 4 Sod
initial-condition parameters ``[p_high, p_low, rho_high, rho_low]`` from
synthetic density observations drawn from the exact Sod solution.

  - ``inference/enkf/`` -- ensemble methods: single-step EnKF, cycled EnKF,
    ES-MDA (all sharing the C++ root filter / ensemble machinery in common.py).
  - ``inference/mcmc/`` -- Bayesian posterior sampling: RW-Metropolis and HMC.
"""
