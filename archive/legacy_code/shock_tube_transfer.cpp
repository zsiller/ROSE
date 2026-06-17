#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>

struct InitialConditions {
    double p_high;
    double p_low;
    double rho_high;
    double rho_low;
    double t;
};

struct ShockState {
    double p_left;
    double dens_left;
    double p_right;
    double dens_right;
    double t;
    double T_left;
    double T_right;
    double a_left;
    double a_right;
    double alpha;
    double P;
    double p_2;
    double dens_2;
    double p_3;
    double V;
    double u2;
    double u3;
    double dens_3;
    double C;
    double x0;
    double xL5;
    double x53;
    double x32;
    double x2R;
    double a2;
    double a3;
};

double f_shock(double x, const ShockState& s) {
    return (std::sqrt(2.0 / (1.4 * (1.4 - 1.0))) * (x - 1.0) / std::sqrt(1.0 + s.alpha * x))
         - ((2.0 / (1.4 - 1.0)) * (s.a_left / s.a_right)
            * (1.0 - std::pow(((s.p_right / s.p_left) * x), ((1.4 - 1.0) / (2.0 * 1.4)))));
}

double findroot(double a, double b, double tol, const ShockState& s) {
    double m;
    while ((b - a) / 2.0 >= tol) {
        m = (a + b) / 2.0;
        const double fm = f_shock(m, s);
        if (fm == 0.0) {
            break;
        } else if (fm * f_shock(a, s) < 0.0) {
            b = m;
        } else {
            a = m;
        }
    }
    return (a + b) / 2.0;
}

ShockState build_shock_state(const InitialConditions& ic) {
    ShockState s{};
    s.p_left = ic.p_high;
    s.dens_left = ic.rho_high;
    s.p_right = ic.p_low;
    s.dens_right = ic.rho_low;
    s.t = ic.t;
    s.x0 = 0.5;

    s.T_left = s.p_left / (s.dens_left * 287.0);
    s.T_right = s.p_right / (s.dens_right * 287.0);
    s.a_left = std::sqrt(1.4 * 287.0 * s.T_left);
    s.a_right = std::sqrt(1.4 * 287.0 * s.T_right);
    s.alpha = (1.4 + 1.0) / (1.4 - 1.0);

    s.P = findroot(2.0, 5.0, 1e-14, s);
    s.p_2 = s.p_right * s.P;
    s.dens_2 = ((1.0 + s.alpha * s.P) / (s.alpha + s.P)) * s.dens_right;
    s.p_3 = s.p_2;
    s.V = (2.0 / (1.4 - 1.0)) * s.a_left
        * (1.0 - std::pow((s.p_3 / s.p_left), ((1.4 - 1.0) / (2.0 * 1.4))));
    s.u2 = s.V;
    s.u3 = s.V;
    s.dens_3 = s.dens_left * std::pow((s.p_3 / s.p_left), (1.0 / 1.4));
    s.C = ((s.P - 1.0) * std::pow(s.a_right, 2)) / (1.4 * s.u2);

    s.xL5 = s.x0 - (s.a_left * s.t);
    s.x53 = s.x0 + ((s.V * (1.4 + 1.0) / 2.0) - s.a_left) * s.t;
    s.x32 = s.x0 + s.V * s.t;
    s.x2R = s.x0 + s.C * s.t;
    s.a2 = std::sqrt(1.4 * (s.p_2 / s.dens_2));
    s.a3 = std::sqrt(1.4 * (s.p_3 / s.dens_3));
    return s;
}

struct RunConfig {
    std::string file_prefix;
    InitialConditions ic;
};

InitialConditions parse_initial_conditions(int argc, char* argv[], int start_idx) {
    if (argc < start_idx + 5) {
        throw std::runtime_error("missing initial condition arguments");
    }

    InitialConditions ic{};
    ic.p_high = std::stod(argv[start_idx + 0]);
    ic.p_low = std::stod(argv[start_idx + 1]);
    ic.rho_high = std::stod(argv[start_idx + 2]);
    ic.rho_low = std::stod(argv[start_idx + 3]);
    ic.t = std::stod(argv[start_idx + 4]);
    return ic;
}

RunConfig parse_args(int argc, char* argv[]) {
    if (argc != 7) {
        throw std::runtime_error(
            "usage: shock_tube_transfer <file_prefix> <p_high> <p_low> <rho_high> <rho_low> <t>"
        );
    }

    RunConfig cfg{};
    cfg.file_prefix = argv[1];
    cfg.ic = parse_initial_conditions(argc, argv, 2);
    return cfg;
}

void write_solution(const ShockState& s, const std::string& file_prefix, int N = 1000) {
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);

    int num_x_L = 0;
    int num_x_5 = 0;
    int num_x_3 = 0;
    int num_x_2 = 0;
    int num_x_R = 0;

    Eigen::VectorXd u5;
    Eigen::VectorXd a5;

    for (double i : x) {
        if (i <= s.xL5) {
            num_x_L++;
        } else if (i <= s.x53) {
            num_x_5++;
            const double u5p = (2.0 / (1.4 + 1.0)) * (((i - s.x0) / s.t) + s.a_left);
            const double a5p = u5p - ((i - s.x0) / s.t);
            u5.conservativeResize(u5.size() + 1);
            u5(u5.size() - 1) = u5p;
            a5.conservativeResize(a5.size() + 1);
            a5(a5.size() - 1) = a5p;
        } else if (i <= s.x32) {
            num_x_3++;
        } else if (i <= s.x2R) {
            num_x_2++;
        } else {
            num_x_R++;
        }
    }

    Eigen::VectorXd p5 = s.p_left * (a5.array() / s.a_left).pow((2.0 * 1.4) / (1.4 - 1.0));
    Eigen::VectorXd dens_5 = (1.4) * p5.array() / (a5.array().square());

    Eigen::VectorXd p_vec_left = Eigen::VectorXd::Constant(num_x_L, s.p_left);
    Eigen::VectorXd p_vec_3 = Eigen::VectorXd::Constant(num_x_3, s.p_3);
    Eigen::VectorXd p_vec_2 = Eigen::VectorXd::Constant(num_x_2, s.p_2);
    Eigen::VectorXd p_vec_right = Eigen::VectorXd::Constant(num_x_R, s.p_right);

    Eigen::VectorXd dens_vec_left = Eigen::VectorXd::Constant(num_x_L, s.dens_left);
    Eigen::VectorXd dens_vec_3 = Eigen::VectorXd::Constant(num_x_3, s.dens_3);
    Eigen::VectorXd dens_vec_2 = Eigen::VectorXd::Constant(num_x_2, s.dens_2);
    Eigen::VectorXd dens_vec_right = Eigen::VectorXd::Constant(num_x_R, s.dens_right);

    Eigen::VectorXd a_vec_left = Eigen::VectorXd::Constant(num_x_L, s.a_left);
    Eigen::VectorXd a_vec_3 = Eigen::VectorXd::Constant(num_x_3, s.a3);
    Eigen::VectorXd a_vec_2 = Eigen::VectorXd::Constant(num_x_2, s.a2);
    Eigen::VectorXd a_vec_right = Eigen::VectorXd::Constant(num_x_R, s.a_right);

    Eigen::VectorXd u_vec_left = Eigen::VectorXd::Constant(num_x_L, 0.0);
    Eigen::VectorXd u_vec_3 = Eigen::VectorXd::Constant(num_x_3, s.u3);
    Eigen::VectorXd u_vec_2 = Eigen::VectorXd::Constant(num_x_2, s.u2);
    Eigen::VectorXd u_vec_right = Eigen::VectorXd::Constant(num_x_R, 0.0);

    Eigen::VectorXd pressure(N);
    Eigen::VectorXd density(N);
    Eigen::VectorXd sos(N);
    Eigen::VectorXd speed(N);

    pressure << p_vec_left, p5, p_vec_3, p_vec_2, p_vec_right;
    density << dens_vec_left, dens_5, dens_vec_3, dens_vec_2, dens_vec_right;
    sos << a_vec_left, a5, a_vec_3, a_vec_2, a_vec_right;
    speed << u_vec_left, u5, u_vec_3, u_vec_2, u_vec_right;
    Eigen::VectorXd momentum = density.array() * speed.array();
    Eigen::VectorXd mach = speed.array() / sos.array();

    const std::string density_path = file_prefix + "_density.txt";
    std::ofstream density_file(density_path);
    density_file << "x-location density-value\n";
    for (int i = 0; i < N; i++) {
        density_file << x[i] << ' ' << density[i] << '\n';
    }
}

int main(int argc, char* argv[]) {
    try {
        const RunConfig cfg = parse_args(argc, argv);
        const ShockState state = build_shock_state(cfg.ic);
        write_solution(state, cfg.file_prefix);
        std::cout << "Wrote " << cfg.file_prefix << "_density.txt\n";
        std::cout << "IC: p_high=" << cfg.ic.p_high
                  << " p_low=" << cfg.ic.p_low
                  << " rho_high=" << cfg.ic.rho_high
                  << " rho_low=" << cfg.ic.rho_low
                  << " t=" << cfg.ic.t << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
