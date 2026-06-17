#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "CDR_1D_model.h"

Params p;

Eigen::SparseMatrix<double> d1 = D1(p);
Eigen::SparseMatrix<double> d2 = D2(p);

Eigen::VectorXd run_cdr_simulation(Params& p, double beta) {
    p.time = p.t_0;
    p.beta = beta;
    Eigen::VectorXd cur = gen_exact(p).col(1);

    while (p.time < p.t_f) {
        p.time += p.dt;
        cur = rk4(cur, p, d1, d2);
    }
    return cur;
}

static double mean(const std::vector<double>& v) {
    if (v.empty()) {
        return 0.0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

/** Sample standard deviation (divide by n - 1). */
static double sample_std_dev(const std::vector<double>& v, double m) {
    if (v.size() < 2) {
        return 0.0;
    }
    double s = 0.0;
    for (double x : v) {
        const double d = x - m;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

int main(int argc, char** argv) {
    constexpr int n = 100;
    std::vector<double> betas(n);
    for (int i = 0; i < n; ++i) {
        betas[i] = i * (10.0 / 49.0);
    }

    // Warm-up (first run can pay cache/branch predictor costs)
    run_cdr_simulation(p, betas[0]);

    std::vector<double> seconds;
    seconds.reserve(betas.size());

    using clock = std::chrono::high_resolution_clock;

    for (double beta : betas) {
        const auto t0 = clock::now();
        run_cdr_simulation(p, beta);
        const auto t1 = clock::now();
        const std::chrono::duration<double> dt = t1 - t0;
        seconds.push_back(dt.count());
    }

    const double total = std::accumulate(seconds.begin(), seconds.end(), 0.0);
    const double m = mean(seconds);
    const double sd = sample_std_dev(seconds, m);

    std::cout << "CDR timing (" << n << " betas, one simulation each)\n";
    std::cout << "  Total wall time:   " << total << " s\n";
    std::cout << "  Mean per run:      " << m << " s\n";
    std::cout << "  Std dev (sample):  " << sd << " s\n";

    return 0;
}
