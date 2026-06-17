#ifndef ENSEMBLE_FAMILY_H_
#define ENSEMBLE_FAMILY_H_

#include <random>
#include <Eigen/Dense>
#include <cstdint>

using namespace Eigen;
using namespace std;

/**
 * @brief Functor that adds Gaussian noise to a scalar value.
 *
 * Seeded at construction; AddGaussian(σ) creates a fresh RNG via
 * std::random_device, while AddGaussian(σ, seed) is reproducible. Used by
 * EnKF::filter() to build the perturbed observation ensemble.
 */
struct AddGaussian {
    mutable std::mt19937_64 gen;
    mutable std::normal_distribution<double> dist;

    explicit AddGaussian(double stddev, std::uint64_t seed = std::random_device{}())
        : gen(seed), dist(0.0, stddev) {}

    double operator()(double x) const { return x + dist(gen); }
};

/**
 * @brief Container the EnKF filter reads: a prebuilt state ensemble plus its
 *        observations, operator, and localization scaffolding.
 *
 * The ensemble is taken VERBATIM (one member per column): the top
 * `num_globals` rows are the augmented global parameters, the rest are local
 * state. Construction injects no noise -- the spread is whatever the members
 * already carry (they come from forward-solved HDF5 files via the Python
 * driver). The augmented-state layout matches enkf_step.cpp / ensemble_common.py.
 */
struct ensembleParams {

public:
    MatrixXd ensemble;   ///< State ensemble: (num_globals + n_local) x ens_size
    VectorXd obs;        ///< Observation vector y^o (size m)
    MatrixXd obs_op;     ///< Observation operator H (t_vars x m), H^T·state -> obs

    vector<VectorXd> obs_loc;   ///< Observation coordinates for localization
    vector<VectorXd> state_loc; ///< State coordinates for localization

    int ens_size;        ///< Number of ensemble members (Ne)

    double loc_rad = 20.0;          ///< Gaspari-Cohn cutoff (cell distance)
    bool use_localization = false;  ///< Apply Schur-product localization in filter()

    double obs_error;        ///< Observation noise std (σ), forms R = σ² I
    long long obs_seed = -1; ///< Seed for obs perturbation; <0 = fresh random each filter()

    int num_globals;     ///< Number of global (augmented) rows at the top of 'ensemble'

    /**
     * @brief Wrap an already-assembled state ensemble together with its
     *        observations and operator.
     *
     * @param prebuilt    State ensemble (t_vars x Ne), one member per column
     * @param observation Observation vector y^o (size m)
     * @param oper        Observation operator H (t_vars x m), H^T·state -> obs
     * @param obs_err     Observation noise std (σ), forms R = σ² I
     * @param n_globals   Number of global (augmented) rows at the top
     *
     * Localization scaffolding (obs_loc/state_loc/loc_rad/use_localization) is
     * set by the caller after construction when localization is enabled.
     */
    ensembleParams(const MatrixXd& prebuilt, const VectorXd& observation,
                   const MatrixXd& oper, const double obs_err, const int n_globals)
    {
        ensemble    = prebuilt;
        ens_size    = static_cast<int>(prebuilt.cols());
        obs         = observation;
        obs_op      = oper;
        obs_error   = obs_err;
        num_globals = n_globals;
    }
};

#endif // ENSEMBLE_FAMILY_H_
