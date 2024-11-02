#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <complex>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef Matrix<complex<double>, Dynamic, Dynamic> MatrixXcd;
typedef Matrix<complex<double>, Dynamic, 1> VectorXcd;

VectorXcd integrate(const MatrixXcd &H, const VectorXcd &psi0, double t_span, double t_eval);
MatrixXcd kroneckerProduct(const MatrixXcd &A, const MatrixXcd &B);
MatrixXcd determine_jump(const vector<MatrixXcd> &c_ops, const VectorXcd &psi, double r2);
vector<VectorXcd> montecarlo(const MatrixXcd &H, const vector<MatrixXcd> &c_ops, const VectorXcd &psi0, const vector<double> &tlist, unsigned int seed = 0);
vector<complex<double>> montecarlo_average(const MatrixXcd &H, const vector<MatrixXcd> &c_ops, const VectorXcd &psi0, const vector<double> &tlist, int ntraj, const MatrixXcd &op);

#endif // FUNCTIONS_H
