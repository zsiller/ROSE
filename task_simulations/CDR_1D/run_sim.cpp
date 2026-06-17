/*
 * run_cdr_1d.cpp
 *
 * This file is the main driver for the 1D Convection-Diffusion-Reaction (CDR) model.
 * It parses command line arguments, sets up the simulation parameters,
 * and writes the output to an HDF5 file.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <H5Cpp.h>

#include "CDR_1D_model.h"

/**
 * @brief Parse a C-string as a double or throw on error.
 *
 * @param s     Null-terminated C-string to parse.
 * @param name  Human-readable name used in error messages.
 * @return Parsed double value.
 *
 * @throws std::runtime_error if the string is missing, not a valid number,
 *         or contains trailing non-numeric characters.
 */
double parse_args(const char* s, const char* name) {

  if (s == nullptr) throw std::runtime_error(std::string("missing argument: ") + name);
  char* end = nullptr;
  const double v = std::strtod(s, &end);

  if (end == s || *end != '\0') {
    std::ostringstream oss;
    oss << "invalid " << name << ": '" << s << "'";
    throw std::runtime_error(oss.str());
  }

  return v;
}
 
/**
 * @brief Write simulation metadata and time-step data to an HDF5 file.
 *
 * The file stores root attributes and:
 * - "x": 1D cell-center locations (length grid_s).
 * - "t": 1D output times (length n_steps).
 * - "u": 2D solution (n_steps x grid_s); u(i,j) is solution at time t(i), point x(j).
 *
 * @param path   Filesystem path to the HDF5 file (overwritten if it exists).
 * @param p      Simulation parameters.
 * @param x      Cell-center locations (size must equal p.grid_s).
 * @param times  Output times (length n_steps).
 * @param u_snapshots Flattened solution: n_steps * grid_s doubles, row-major (step 0, then step 1, ...).
 * @param steps_between_snapshots Number of time steps between each saved snapshot (e.g. 100).
 *
 * @throws std::runtime_error on size mismatch or HDF5 failure.
 */
void write_hdf5(
    const std::string& path,
    const Params& p,
    const Eigen::VectorXd& x,
    const std::vector<double>& times,
    const std::vector<double>& u_snapshots,
    int steps_between_snapshots) {

  const hsize_t n_steps = static_cast<hsize_t>(times.size());
  if (static_cast<int>(x.size()) != p.grid_s) {
    throw std::runtime_error("write_hdf5: x size mismatch");
  }
  if (u_snapshots.size() != n_steps * static_cast<hsize_t>(p.grid_s)) {
    throw std::runtime_error("write_hdf5: u_snapshots size mismatch");
  }

  H5::H5File file(path, H5F_ACC_TRUNC);

  // Root attributes
  {
    const H5::DataSpace scalar(H5S_SCALAR);
    auto write_attr_double = [&](const char* key, double value) {
      H5::Attribute a = file.createAttribute(key, H5::PredType::NATIVE_DOUBLE, scalar);
      a.write(H5::PredType::NATIVE_DOUBLE, &value);
    };
    auto write_attr_int = [&](const char* key, int value) {
      H5::Attribute a = file.createAttribute(key, H5::PredType::NATIVE_INT, scalar);
      a.write(H5::PredType::NATIVE_INT, &value);
    };

    write_attr_int("grid_s", p.grid_s);
    write_attr_int("n_steps", static_cast<int>(n_steps));
    write_attr_int("steps_between_snapshots", steps_between_snapshots);
    write_attr_double("dt", p.dt);
    write_attr_double("t_final", p.t_f);
    write_attr_double("beta", p.beta);
    write_attr_double("a", p.a);
    write_attr_double("mu", p.mu);
  }

  // Dataset: x (1D)
  const hsize_t dims_x[1] = {static_cast<hsize_t>(p.grid_s)};
  H5::DataSpace space_x(1, dims_x);
  H5::DataSet ds_x = file.createDataSet("x", H5::PredType::NATIVE_DOUBLE, space_x);
  ds_x.write(x.data(), H5::PredType::NATIVE_DOUBLE);

  // Dataset: t (1D)
  const hsize_t dims_t[1] = {n_steps};
  H5::DataSpace space_t(1, dims_t);
  H5::DataSet ds_t = file.createDataSet("t", H5::PredType::NATIVE_DOUBLE, space_t);
  ds_t.write(times.data(), H5::PredType::NATIVE_DOUBLE);

  // Dataset: u (2D: n_steps x grid_s)
  const hsize_t dims_u[2] = {n_steps, static_cast<hsize_t>(p.grid_s)};
  H5::DataSpace space_u(2, dims_u);
  H5::DataSet ds_u = file.createDataSet("u", H5::PredType::NATIVE_DOUBLE, space_u);
  ds_u.write(u_snapshots.data(), H5::PredType::NATIVE_DOUBLE);
}
 
/**
 * @brief Parse a C-string as a positive integer (number of time steps).
 */
int parse_steps(const char* s, const char* name) {
  if (s == nullptr) throw std::runtime_error(std::string("missing argument: ") + name);
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v < 1) {
    std::ostringstream oss;
    oss << "invalid " << name << " (must be a positive integer): '" << s << "'";
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

/**
 * @brief Command-line driver for the 1D CDR model.
 *
 * Writes the solution at step 0 and then every write_every_n_steps time steps
 * into a single HDF5 file.
 *
 * Usage: run_cdr_1d <t_final> <beta> <write_every_n_steps>
 *   write_every_n_steps: write a snapshot every this many time steps (e.g. 100).
 *
 * @return 0 on success, 2 on invalid usage, 1 on runtime error.
 */
int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "Usage:\n"
                << "  " << argv[0] << " <data_path> <t_final> <beta> <write_every_n_steps>\n\n"
                << "  data_path:            path to save the data\n"
                << "  t_final:              final simulation time\n"
                << "  beta:                 reaction coefficient\n"
                << "  write_every_n_steps:  write snapshot every N time steps (e.g. 100)\n\n";
      return 2;
    }
    const std::string data_path = argv[1];
    Params p;
    p.t_f = parse_args(argv[2], "t_final");
    p.beta = parse_args(argv[3], "beta");
    const int write_every_n_steps = parse_steps(argv[4], "write_every_n_steps");

    Eigen::SparseMatrix<double> d1 = D1(p);
    Eigen::SparseMatrix<double> d2 = D2(p);

    p.time = p.t_0;
    Eigen::MatrixXd exact = gen_exact(p);
    Eigen::VectorXd x = exact.col(0);
    Eigen::VectorXd cur = exact.col(1);

    std::vector<double> out_times;
    std::vector<double> u_snapshots;

    auto save_snapshot = [&]() {
      out_times.push_back(p.time);
      for (int i = 0; i < cur.size(); ++i) {
        u_snapshots.push_back(cur(i));
      }
    };

    save_snapshot();  // initial condition (step 0)
    int step = 0;
    while (p.time < p.t_f) {
      p.time += p.dt;
      cur = rk4(cur, p, d1, d2);
      ++step;
      if (step % write_every_n_steps == 0) {
        save_snapshot();
      }
    }
    // Save final state if we did not already save on the last step
    if (step % write_every_n_steps != 0) {
      save_snapshot();
    }

    write_hdf5(data_path, p, x, out_times, u_snapshots, write_every_n_steps);

    std::cout << "Wrote data to: " << data_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
