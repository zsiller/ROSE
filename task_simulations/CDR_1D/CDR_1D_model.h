// This file contains the definitions of the functions and classes for the 1D CDR model.
/*
 * CDR_1D_model.h
 *
 * This header defines the key data structures and function prototypes
 * for the 1D Convection-Diffusion-Reaction (CDR) model.
 *
 * Overview:
 * - Contains the Params struct, which specifies all simulation parameters.
 * - Declares functions for generating analytic solutions, building spatial
 *   derivative operators, and performing time integration.
 * - Relies on Eigen for vector/matrix storage and linear algebra routines.
 *
 * Key Components:
 * 
 * struct Params
 *   - Holds all configurable model settings (grid size, time step, coefficients,
 *     boundary conditions, etc).
 * 
 * MatrixXd gen_exact(Params& p)
 *   - Computes the exact analytic solution for the current problem setup, storing
 *     values at each spatial grid point.
 * 
 * SparseMatrix<double> D1(Params& p)
 *   - Constructs the first-order spatial derivative operator (finite difference).
 *
 * SparseMatrix<double> D2(Params& p)
 *   - Constructs the second-order spatial derivative operator (finite difference).
 * 
 * VectorXd spacial_var(Params& p, const double t)
 *   - Computes the spatially (and temporally) varying solution profile.
 * 
 * VectorXd rk4(const VectorXd&, Params&, const SparseMatrix<double>&, const SparseMatrix<double>&)
 *   - Advances the numerical solution in time by one step using the classic 4th order Runge-Kutta.
 * 
 * Notes:
 * - This header assumes Eigen's dense and sparse matrix headers are available.
 * - All Eigen objects use double precision.
 * - By convention, all function implementations are provided in CDR_1D_model.cpp.
 */


#ifndef CDR_1D_MODEL_H_
#define CDR_1D_MODEL_H_

#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Sparse>
#include <vector>
#include <cstdlib>
#include <sstream>


using namespace Eigen;
using namespace std;


/**
 * @brief This struct contains the parameters for the 1D CDR model.
 * 
 * @param grid_s The number of grid points in the spatial domain.
 * @param a The reaction rate coefficient.
 * @param mu The diffusion rate coefficient.
 * @param beta The reaction rate coefficient.
 * @param dt The time step size.
 * @param time The current time.
 * @param dx The spatial step size.
 * @param x_0 The left boundary of the spatial domain.
 * @param x_f The right boundary of the spatial domain.
 * @param t_0 The initial time.
 * @param t_f The final time.
 * 
 * The parameters are set to default values in the struct declaration.
 */
struct Params {

  int grid_s = 128;
  double a = 1.0;
  double mu = 1.0;
  double beta = 1.0;
  double dt = pow(10,-5);
  double time = 0.0;
  double dx = 1.0/grid_s;

  double x_0 = 0.0;
  double x_f = 1.0;

  double t_0 = 0.0;
  double t_f = .25;

};

MatrixXd gen_exact(Params& p);

SparseMatrix<double> D1(Params& p);

SparseMatrix<double> D2(Params& p);

VectorXd spacial_var(Params& p, const double t);

VectorXd rk4(const VectorXd& current, Params& p, const SparseMatrix<double>& D1, const SparseMatrix<double>& D2);

#endif // CDR_1D_MODEL_H_
