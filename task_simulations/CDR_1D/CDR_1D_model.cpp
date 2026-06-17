/*
 * CDR_1D_model.cpp
 *
 * Implementation file for the 1D Convection-Diffusion-Reaction (CDR) model.
 *
 * This file contains the definition of core functions that simulate the CDR equation
 * in one spatial dimension. These functions rely on Eigen for linear algebra, and
 * implement generation of analytic solutions, construction of finite difference spatial
 * operators (first and second derivatives), as well as classical Runge-Kutta temporal
 * integration for advancing the PDE solution in time.
 *
 * Functions implemented in this file:
 *   - gen_exact(Params& p): Build the analytic/exact solution over the spatial grid.
 *   - D1(Params& p): Construct the sparse matrix for the first-order spatial derivative
 *                    using a finite-difference stencil.
 *   - D2(Params& p): Construct the sparse matrix for the second-order spatial derivative
 *                    using a finite-difference stencil.
 *   - spacial_var(Params& p, const double t): Generate a spatially/temporally varying
 *                    solution profile (details in implementation).
 *   - rk4(const VectorXd& cur, Params& p, const SparseMatrix<double>& D1,
 *         const SparseMatrix<double>& D2): Advance the solution one time step using
 *                    classic RK4 method, given current state and operator matrices.
 *
 * The Params struct (see header) encapsulates all of the settings for grid size,
 * step sizes, boundary locations, and PDE coefficients. Most functions take it as an input.
 *
 * This implementation assumes solution vectors and system matrices are organized
 * to correspond to cell-centered finite difference grid ordering.
 *
 * Note: The boundary conditions and stencils are hardcoded for a typical 1D grid.
 * Adjust stencils in D1 and D2 if different accuracy orders or grid/bc types are needed.
 *
 * See CDR_1D_model.h for further details on arguments and usage.
 */


#include "CDR_1D_model.h"


/**
 * @brief Generate the analytic solution u(x, t) = x^2 e^{-t} on the current grid.
 *
 * @param p Problem parameters containing grid size, spacing, and current time.
 * @return Matrix with two columns: cell-center locations x and exact solution values u(x, p.time).
 */
MatrixXd gen_exact(Params& p) {

  MatrixXd mat_exact(p.grid_s, 2);
  
  double x_c = p.x_0 + p.dx/2.0;

  for (int i=0; i<mat_exact.rows(); ++i) {
    mat_exact(i,0) = x_c;
    mat_exact(i,1) = pow(x_c,2.0)*exp(-p.time);
    x_c+=p.dx;
  }

  return mat_exact;
}

/**
 * @brief Construct first-order derivative operator using a fourth-order finite-difference stencil.
 *
 * Uses centered fourth-order accurate differences in the interior and one-sided
 * fourth-order formulas near the boundaries. Coefficients are assembled into
 * a sparse matrix compatible with Eigen's sparse linear algebra routines.
 *
 * @param p Problem parameters providing grid size and spacing.
 * @return Sparse matrix approximating \f$\partial / \partial x\f$ on the 1D grid.
 */
SparseMatrix<double> D1(Params& p){

  typedef Triplet<double> T;
  vector<T> triplets;

  for (int i = 0; i < p.grid_s; ++i) {

    // Interior nodes: centered 4th-order stencil.
    if (1 < i and i < (p.grid_s - 2)) {
      triplets.emplace_back(i, i + 2, - 1.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 1, + 8.0/(12.0*p.dx));
      triplets.emplace_back(i, i - 1, - 8.0/(12.0*p.dx));
      triplets.emplace_back(i, i - 2, + 1.0/(12.0*p.dx));
    }

    // Left boundary: 4th-order one-sided stencil at first node.
    else if (i == 0) {
      triplets.emplace_back(i, i, - 10.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 1, + 18.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 2, - 6.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 3, + 1.0/(12.0*p.dx));
    }

    // Left-near-boundary node: modified one-sided stencil.
    else if (i == 1) {
      triplets.emplace_back(i, i, - 10.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 1, + 18.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 2, - 6.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 3, + 1.0/(12.0*p.dx));
      triplets.emplace_back(i, i - 1, - 3.0/(12.0*p.dx));
    }

    // Right-near-boundary node: symmetric to i == 1 case.
    else if (i == (p.grid_s - 2) ) {
      triplets.emplace_back(i, i, + 10.0/(12.0*p.dx));
      triplets.emplace_back(i, i - 1, - 18.0/(12.0*p.dx));
      triplets.emplace_back(i, i - 2, + 6.0/(12.0*p.dx));
      triplets.emplace_back(i, i - 3, - 1.0/(12.0*p.dx));
      triplets.emplace_back(i, i + 1, + 3.0/(12.0*p.dx));

    }

    // Right boundary: coefficients chosen so that the last row remains consistent
    // with the interior stencil while satisfying the one-sided approximation.
    else {
      triplets.emplace_back(i, i, (10.0 - 10.0)/(12.0*p.dx));
      triplets.emplace_back(i, i - 1, (-18.0 + 18.0)/(12.0*p.dx));
      triplets.emplace_back(i, i - 2, (6.0 - 6.0)/(12.0*p.dx));
      triplets.emplace_back(i, i - 3, (-1.0 + 1.0)/(12.0*p.dx));
    }
  }

  SparseMatrix<double> D_first(p.grid_s, p.grid_s);
  D_first.setFromTriplets(triplets.begin(), triplets.end());
  D_first.makeCompressed();

  return D_first;
}

/**
 * @brief Construct second-order derivative operator using a fourth-order finite-difference stencil.
 *
 * Uses centered fourth-order accurate differences in the interior and appropriate
 * one-sided fourth-order formulas near the boundaries. The result is a sparse
 * matrix approximating \f$\partial^2 / \partial x^2\f$.
 *
 * @param p Problem parameters providing grid size and spacing.
 * @return Sparse matrix approximating the second spatial derivative on the grid.
 */
SparseMatrix<double> D2(Params& p) {

typedef Triplet<double> T;
  vector<T> triplets;

  for (int i = 0; i < p.grid_s; ++i) {

    // Interior nodes: centered 4th-order stencil for d^2/dx^2.
    if (1 < i and i < (p.grid_s - 2)) {
      triplets.emplace_back(i, i, -30.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 2, -1.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 1, 16.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 1, 16.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 2, -1.0/(12.0*pow(p.dx,2.0)));
    }

    // Left boundary: 4th-order one-sided stencil at first node.
    else if (i == 0) {
      triplets.emplace_back(i, i, -15.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 1, -4.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 2, 14.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 3, -6.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 4, 1.0/(12.0*pow(p.dx,2.0)));
    }

    // Left-near-boundary node: modified one-sided stencil.
    else if (i == 1) {
      triplets.emplace_back(i, i, -15.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 1, -4.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 2, 14.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 3, -6.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 4, 1.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 1, 10.0/(12.0*pow(p.dx,2.0)));
    }

    // Right-near-boundary node: symmetric to i == 1 case.
    else if (i == (p.grid_s - 2) ) {
      triplets.emplace_back(i, i, -15.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 1, -4.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 2, 14.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 3, -6.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 4, 1.0/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i + 1, 10.0/(12.0*pow(p.dx,2.0)));
    }

    // Right boundary: coefficients adjusted to preserve overall accuracy and stability.
    else {
      triplets.emplace_back(i, i, (-15.0 - (100.0/3.0))/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 1, (-4.0 + 60)/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 2, (14.0 - 20.0)/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 3, (-6.0 + (10.0/3.0))/(12.0*pow(p.dx,2.0)));
      triplets.emplace_back(i, i - 4, 1.0/(12.0*pow(p.dx,2.0)));
    }
  }

  SparseMatrix<double> D_sec(p.grid_s, p.grid_s);
  D_sec.setFromTriplets(triplets.begin(), triplets.end());
  D_sec.makeCompressed();

  return D_sec;
}

/**
 * @brief Compute spatially varying source term as a function of position and time.
 *
 * The profile is defined in normalized coordinates on \[0, 1\] and scaled so that
 * it decays exponentially in time like \f$e^{-t}\f$.
 *
 * @param p Problem parameters providing grid size and spacing.
 * @param t Time at which the source profile is evaluated.
 * @return Vector of source values at all grid cell centers.
 */
VectorXd spacial_var(Params& p, const double t) {

  VectorXd var(p.grid_s);
  double loc;

  for (int i = 0; i < var.size(); ++i) {
    loc = (double)i/(double)p.grid_s + p.dx/2.0;
    var(i) = 2*(loc - 1.0)*exp(-t);
  }

  return var;
}

/**
 * @brief Advance the CDR solution one time step using classical RK4.
 *
 * The spatial derivative operators D1 and D2 are assumed fixed over the step.
 * Boundary conditions are enforced by modifying the spatial derivatives near
 * the right boundary inside the time-derivative lambda.
 *
 * @param current Current solution vector at time p.time.
 * @param p Problem parameters including coefficients and time step size.
 * @param D1 First-derivative operator matrix.
 * @param D2 Second-derivative operator matrix.
 * @return Solution vector advanced by one time step.
 */
VectorXd rk4(const VectorXd& current, Params& p, const SparseMatrix<double>& D1, const SparseMatrix<double>& D2) {

  // Time-derivative operator for the semi-discrete CDR system du/dt = F(u, t).
  auto solve = [&](const VectorXd& current, const double t) {

    VectorXd dx1 = D1 * current;
    VectorXd dx2 = D2 * current;

    dx1(p.grid_s - 1) += 2.0*exp(-t);
    dx2(p.grid_s - 1) += 20.0*exp(-t)/(p.dx*3.0);

    VectorXd spacial = -p.a * dx1 + p.mu * dx2 - p.beta*current + spacial_var(p,t);

    return spacial;
  };

  double t = p.time;

  VectorXd k_1 = solve(current, t).eval();
  VectorXd k_2 = solve(current + k_1*p.dt/2.0, t + p.dt/2.0).eval();
  VectorXd k_3 = solve(current + k_2*p.dt/2.0, t + p.dt/2.0).eval();
  VectorXd k_4 = solve(current + k_3*p.dt, t + p.dt);

  return current + (k_1 + 2.0*k_2 + 2.0*k_3 + k_4)*p.dt/6.0;
}
