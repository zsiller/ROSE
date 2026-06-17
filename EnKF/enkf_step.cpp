// One EnKF analysis step as a standalone executable, for use as the root filter
// behind a Python driver (e.g. task_simulations/Shock_Tube/cycle_enkf.py). Python
// manages the cycle loop and builds the observation operator; this program does
// the matrix-heavy analysis in C++/Eigen by running the reference EnKF::filter().
//
// It reads one little-endian binary bundle (path = argv[1]) and writes the
// analysis ensemble back as raw doubles (path = argv[2]). All matrices are
// column-major to match Eigen's default storage, so Python writes Fortran order.
//
// Input layout:
//   int32   n_state, ne, m, num_globals, localize
//   int64   seed                 (obs-perturbation seed; <0 = fresh random)
//   double  obs_error, loc_rad
//   double  ensemble[n_state*ne] (column-major)
//   double  obs[m]
//   double  H[n_state*m]         (column-major; H^T @ state -> obs space)
//   double  state_loc[n_state]   (per-row cell position; globals ignored)
//   double  obs_loc[m]           (per-obs cell position)
// Output layout:
//   double  analysis[n_state*ne] (column-major)
//
// Optional diagnostics: if a 4th argument (argv[3]) is given, the perturbed
// observation ensemble y^o_e (m*ne doubles, column-major) is also written there
// -- one independently perturbed copy of y^o per member, for confirming that
// the stochastic filter perturbs observations per member.
//
// Build (from repo root):
//   g++ -O3 -std=c++17 -I/usr/include/eigen3 EnKF/enkf_step.cpp -o EnKF/enkf_step

#include "EnKF.h"

#include <Eigen/Dense>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;

template <class T>
static void rd(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: enkf_step <in.bin> <out.bin>\n";
        return 1;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::cerr << "enkf_step: cannot open " << argv[1] << "\n";
        return 1;
    }

    int32_t n_state = 0, ne = 0, m = 0, num_globals = 0, localize = 0;
    int64_t seed = -1;
    double obs_error = 0.0, loc_rad = 0.0;
    rd(in, n_state); rd(in, ne); rd(in, m); rd(in, num_globals); rd(in, localize);
    rd(in, seed); rd(in, obs_error); rd(in, loc_rad);

    MatrixXd ensemble(n_state, ne);
    in.read(reinterpret_cast<char*>(ensemble.data()), sizeof(double) * n_state * ne);
    VectorXd obs(m);
    in.read(reinterpret_cast<char*>(obs.data()), sizeof(double) * m);
    MatrixXd H(n_state, m);
    in.read(reinterpret_cast<char*>(H.data()), sizeof(double) * n_state * m);
    VectorXd state_loc_v(n_state), obs_loc_v(m);
    in.read(reinterpret_cast<char*>(state_loc_v.data()), sizeof(double) * n_state);
    in.read(reinterpret_cast<char*>(obs_loc_v.data()), sizeof(double) * m);
    if (!in) {
        std::cerr << "enkf_step: short read on " << argv[1] << "\n";
        return 1;
    }
    in.close();

    // Wrap the bundle into the container EnKF::filter reads.
    ensembleParams ep(ensemble, obs, H, obs_error, num_globals);

    std::vector<VectorXd> sloc(n_state, VectorXd::Zero(1));
    std::vector<VectorXd> oloc(m, VectorXd::Zero(1));
    for (int i = 0; i < n_state; ++i) sloc[i](0) = state_loc_v(i);
    for (int j = 0; j < m; ++j) oloc[j](0) = obs_loc_v(j);
    ep.state_loc        = sloc;
    ep.obs_loc          = oloc;
    ep.loc_rad          = loc_rad;
    ep.use_localization = (localize != 0);
    ep.obs_seed         = seed;

    EnKF enkf(ep);
    enkf.filter();

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        std::cerr << "enkf_step: cannot write " << argv[2] << "\n";
        return 1;
    }
    out.write(reinterpret_cast<const char*>(enkf.ens_params.ensemble.data()),
              sizeof(double) * n_state * ne);
    out.close();

    // Optional: dump the perturbed observation ensemble (m x ne) for diagnostics.
    if (argc >= 4) {
        std::ofstream pout(argv[3], std::ios::binary);
        if (!pout) {
            std::cerr << "enkf_step: cannot write " << argv[3] << "\n";
            return 1;
        }
        const MatrixXd& po = enkf.perturbed_obs();
        pout.write(reinterpret_cast<const char*>(po.data()), sizeof(double) * m * ne);
        pout.close();
    }
    return 0;
}
