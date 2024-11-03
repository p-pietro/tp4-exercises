#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Eigen/Dense>
#include <complex>
#include <vector>

using namespace Eigen;
using namespace std;

typedef Matrix<complex<double>, Dynamic, Dynamic> MatrixXcd;
typedef Matrix<complex<double>, Dynamic, 1> VectorXcd;

VectorXcd integrate(const MatrixXcd &H, const VectorXcd &psi0, double t_span,
                    double t_eval, bool use_ode = true);
MatrixXcd kroneckerProduct(const MatrixXcd &A, const MatrixXcd &B);
VectorXcd vkroneckerProduct(const VectorXcd &A, const VectorXcd &B);
MatrixXcd determine_jump(const vector<MatrixXcd> &c_ops, const VectorXcd &psi,
                         double r2);
vector<VectorXcd> montecarlo(const MatrixXcd &H, const vector<MatrixXcd> &c_ops,
                             const VectorXcd &psi0, const vector<double> &tlist,
                             unsigned int seed = 0, bool use_ode = true);
vector<complex<double>> montecarlo_average(
    const MatrixXcd &H, const vector<MatrixXcd> &c_ops, const VectorXcd &psi0,
    const vector<double> &tlist, size_t ntraj, const MatrixXcd &op,
    unsigned int seed = 0, bool use_ode = true);

#endif  // FUNCTIONS_H
