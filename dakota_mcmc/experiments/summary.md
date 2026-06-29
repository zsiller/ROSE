# MCMC method-comparison summary

| run | group | status | N | acc (chain) | acc (QUESO) | post RMSE | runtime |
|---|---|---|---|---|---|---|---|
| am__ens_file | core | ok | 12000 | 0.125 | 0.114 | 1.14e-02 | 5522 |
| am__prior_native | core | ok | 12000 | 0.095 | 0.090 | 1.16e-02 | 5533 |
| am__prior_std | core | ok | 12000 | 0.087 | 0.075 | 1.15e-02 | 5510 |
| dr__ens_file | core | ok | 15000 | 0.228 | 0.228 | 1.16e-02 | 12465 |
| dr__prior_native | core | ok | 15000 | 0.021 | 0.022 | 1.15e-02 | 13808 |
| dr__prior_std | core | ok | 15000 | 0.024 | 0.025 | 1.15e-02 | 13809 |
| dram__ens_file | core | ok | 12000 | 0.717 | 0.695 | 1.14e-02 | 10058 |
| dram__prior_native | core | ok | 12000 | 0.712 | 0.687 | 1.13e-02 | 10428 |
| dram__prior_std | core | ok | 12000 | 0.734 | 0.711 | 1.19e-02 | 10432 |
| mh__ens_file | core | ok | 15000 | 0.013 | 0.014 | 1.09e-02 | 6893 |
| mh__prior_native | core | ok | 15000 | 0.001 | 0.001 | 1.32e-02 | 6937 |
| mh__prior_std | core | ok | 15000 | 0.001 | 0.001 | 1.27e-02 | 6937 |
| deriv_dram_std | gradient | ok | 6000 | 0.729 | 0.702 | 1.18e-02 | 3987 |
| deriv_mh_native | gradient | ok | 6000 | 0.370 | 0.373 | 1.12e-02 | 1891 |
| deriv_mh_std | gradient | ok | 6000 | 0.361 | 0.365 | 1.14e-02 | 2739 |
| scale_mh_m0.001 | scale | ok | 10000 | 0.400 | 0.406 | 1.17e-02 | 4596 |
| scale_mh_m0.005 | scale | ok | 10000 | 0.155 | 0.157 | 1.16e-02 | 4579 |
| scale_mh_m0.02 | scale | ok | 10000 | 0.046 | 0.047 | 1.14e-02 | 4622 |
| scale_mh_m0.1 | scale | ok | 10000 | 0.007 | 0.008 | 1.10e-02 | 4613 |
