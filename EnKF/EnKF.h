#ifndef ENKF_H_
#define ENKF_H_

#include "Ensemble_Family.h"
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace Eigen;
using namespace std;

/**
 * @brief Basic Ensemble Kalman Filter.
 *
 * Conventions:
 *   - State ensemble X: (n_state x Ne), stored in ens_params.ensemble.
 *   - Observation operator H: ens_params.obs_op.
 *   - Observations y^o: ens_params.obs (m).
 *
 * Update performed:
 *   X^a = X^f + PH^T S^{-1} ( y^o_e - H X^f ),
 *   S = H P H^T + R.
 * 
 * Notes: 
 *  - Instability and ensemble collapse as number of observations increases.
 *  
 * Future Changes:
 *  - Replace pseudo inverses
 */

class EnKF {

public:

    ensembleParams ens_params;

    EnKF(ensembleParams& ensP) : ens_params(ensP) {
         VectorXd sigma(ens_params.obs.size());
        sigma.fill(ens_params.obs_error);
        obs_covar = sigma.array().square().matrix().asDiagonal();
    }

    
    /// One EnKF analysis using perturbed observations
    void filter() {

        // Background covariance P (n x n)
        state_covar = covar(ens_params.ensemble);
        
        // Perturbed observations y^o_e (R = obs_error^2 I, diagonal)
        o_error = obs_error();

        // Kalman gain
        kalman = kalman_gain();
        
        // Update ensemble with current ensemble and kalman gain and innovation
        ens_params.ensemble.noalias() = ens_params.ensemble + kalman * innovation(o_error);

        }

    /// Perturbed observations y^o_e (m x Ne) built by the last filter() call --
    /// each column an independently perturbed copy of y^o. Exposed for diagnostics.
    const MatrixXd& perturbed_obs() const { return o_error; }

private:

    MatrixXd state_covar;
    MatrixXd obs_covar;

    MatrixXd o_error;

    MatrixXd kalman;

    // ----- helpers -----

    /**
     * @brief Sample covariance of ensemble.
     * Returns (n x n).
     */
    MatrixXd covar(const MatrixXd mat) {
        double n = 1.0 / (ens_params.ens_size - 1.0);

        VectorXd bar = mat.rowwise().mean();
        
        MatrixXd mat_dif = mat - bar.replicate(1, ens_params.ens_size);

        return n*(mat_dif*(mat_dif.transpose()));
    }

    /**
     * @brief Generate perturbed observations y^o_e = y^o + ε, ε ~ N(0, R) per column.
     * Return matrix of perturbed observations
     */
    MatrixXd obs_error() {

        MatrixXd obs_e = (ens_params.obs.replicate(1, ens_params.ens_size));

        if (ens_params.obs_seed >= 0)
            return obs_e.unaryExpr(AddGaussian(ens_params.obs_error,
                                               static_cast<std::uint64_t>(ens_params.obs_seed)));
        return obs_e.unaryExpr(AddGaussian(ens_params.obs_error));
    }

    /**
     * @brief Gaspari-Cohn 5th-order localization weight for a separation `dist`
     *        with compact support cutoff `radius` (weight is 0 for dist>=radius).
     *
     * Smoothly tapers from 1 at zero separation to 0 at the cutoff, so it can be
     * used as a Schur (element-wise) multiplier to damp spurious long-range
     * sample covariances. (Gaspari & Cohn 1999, eq. 4.10, with c = radius/2.)
     */
    static double gaspari_cohn(double dist, double radius) {
        if (radius <= 0.0) return 1.0;
        const double c = radius / 2.0;
        const double r = std::abs(dist) / c;
        if (r >= 2.0) return 0.0;
        if (r <= 1.0) {
            return -0.25 * pow(r, 5) + 0.5 * pow(r, 4) + 0.625 * pow(r, 3)
                   - (5.0 / 3.0) * pow(r, 2) + 1.0;
        }
        return (1.0 / 12.0) * pow(r, 5) - 0.5 * pow(r, 4) + 0.625 * pow(r, 3)
               + (5.0 / 3.0) * pow(r, 2) - 5.0 * r + 4.0 - (2.0 / 3.0) / r;
    }

    /**
     * @brief State-by-observation localization matrix C_xy (n x m).
     *
     * Local state rows are tapered by their grid distance to each observation;
     * the leading `num_globals` (augmented) rows have no spatial location, so
     * they are left untapered (weight 1) and keep their full, all-observation
     * influence.
     */
    MatrixXd state_obs_loc() {
        const int n = static_cast<int>(ens_params.ensemble.rows());
        const int m = static_cast<int>(ens_params.obs.size());
        MatrixXd C(n, m);
        for (int i = 0; i < n; ++i) {
            const bool is_global = (i < ens_params.num_globals);
            for (int j = 0; j < m; ++j) {
                if (is_global) { C(i, j) = 1.0; continue; }
                const double d = ens_params.state_loc[i](0) - ens_params.obs_loc[j](0);
                C(i, j) = gaspari_cohn(d, ens_params.loc_rad);
            }
        }
        return C;
    }

    /**
     * @brief Observation-by-observation localization matrix C_yy (m x m),
     *        tapering the innovation covariance by inter-observation distance.
     */
    MatrixXd obs_obs_loc() {
        const int m = static_cast<int>(ens_params.obs.size());
        MatrixXd C(m, m);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                const double d = ens_params.obs_loc[i](0) - ens_params.obs_loc[j](0);
                C(i, j) = gaspari_cohn(d, ens_params.loc_rad);
            }
        }
        return C;
    }

    /**
     * @brief Generate kalman gain matrix from covariances and Y (ensemble in observation space).
     * Return matrix of perturbed observations
     */
    MatrixXd kalman_gain() {

        // Cross covariance P H (n x m) and innovation covariance H^T P H (m x m).
        MatrixXd PH  = state_covar * ens_params.obs_op;
        MatrixXd HPH = ens_params.obs_op.transpose() * PH;

        // Schur-product localization: taper spurious long-range sample covariances.
        if (ens_params.use_localization) {
            PH  = PH.cwiseProduct(state_obs_loc());
            HPH = HPH.cwiseProduct(obs_obs_loc());
        }

        MatrixXd inv = obs_covar + HPH;

        return PH * inv.completeOrthogonalDecomposition().pseudoInverse();
    }

    /**
     * @brief Compute difference between perturbed observations and ensemble.
     * Return matrix of innovation
     */
    MatrixXd innovation(const MatrixXd perturbed_obs) {
        
        MatrixXd state_rows = ens_params.obs_op.transpose() * ens_params.ensemble;
        
        return perturbed_obs - state_rows;
    }
};
#endif // ENKF_H_