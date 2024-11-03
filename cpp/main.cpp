#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <matplot/matplot.h>

#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

#include "functions.h"

using namespace Eigen;
using namespace std;
using namespace std::chrono;

typedef Matrix<complex<double>, Dynamic, Dynamic> MatrixXcd;
typedef Matrix<complex<double>, Dynamic, 1> VectorXcd;

struct Params {
  MatrixXcd H;
};

/*
int schrodinger_eq(double t, const double y[], double f[], void *params) {
    Params *p = (Params *)params;
    Map<const VectorXcd> psi(reinterpret_cast<const complex<double>*>(y),
p->H.rows()); Map<VectorXcd> dpsi_dt(reinterpret_cast<complex<double>*>(f),
p->H.rows()); dpsi_dt = -1i * p->H * psi; return GSL_SUCCESS;
}
*/

/*
VectorXcd integrate(const MatrixXcd &H, const VectorXcd &psi0, double t_span,
double t_eval) { gsl_odeiv2_system sys = {schrodinger_eq, nullptr, H.rows() * 2,
new Params{H}}; gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys,
gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0); VectorXcd psi = psi0; double t = 0.0;
    double y[psi.size() * 2];
    for (int i = 0; i < psi.size(); ++i) {
        y[2 * i] = real(psi[i]);
        y[2 * i + 1] = imag(psi[i]);
    }
    gsl_odeiv2_driver_apply(d, &t, t_eval, y);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = complex<double>(y[2 * i], y[2 * i + 1]);
    }
    gsl_odeiv2_driver_free(d);
    delete (Params *)sys.params;
    return psi;
}
*/

int main() {
  const int N = 2;
  const double wc = 1.0e-3 * 2 * M_PI;
  const double wa = wc;
  const double g = 0.05 * 2 * M_PI;
  const double kappa = 0.005;
  const double gamma = 0.05;
  const double n_th_a = 0.1;

  VectorXcd psi0(4);
  psi0 << 0, 1, 0, 0;

  MatrixXcd a = MatrixXcd::Zero(N, N);
  for (int n = 1; n < N; ++n) {
    a(n - 1, n) = sqrt(n);
  }
  a = kroneckerProduct(a, MatrixXcd::Identity(2, 2)).eval();

  MatrixXcd a_dag = MatrixXcd::Zero(N, N);
  for (int n = 0; n < N - 1; ++n) {
    a_dag(n + 1, n) = sqrt(n + 1);
  }
  a_dag = kroneckerProduct(a_dag, MatrixXcd::Identity(2, 2)).eval();

  MatrixXcd sm = MatrixXcd::Zero(2, 2);
  sm(0, 1) = 1;
  sm = kroneckerProduct(MatrixXcd::Identity(N, N), sm).eval();

  MatrixXcd sp = MatrixXcd::Zero(2, 2);
  sp(1, 0) = 1;
  sp = kroneckerProduct(MatrixXcd::Identity(N, N), sp).eval();

  MatrixXcd sz = MatrixXcd::Zero(2, 2);
  sz(0, 0) = 1;
  sz(1, 1) = -1;
  sz = kroneckerProduct(MatrixXcd::Identity(N, N), sz).eval();

  MatrixXcd H = wc * a_dag * a + wa / 2 * sz + g * (a_dag * sm + a * sp);

  vector<MatrixXcd> c_op_list;
  c_op_list.push_back(sqrt(kappa * (1 + n_th_a)) * a);
  c_op_list.push_back(sqrt(kappa * n_th_a) * a_dag);
  c_op_list.push_back(sqrt(gamma * (1 + n_th_a)) * sm);
  c_op_list.push_back(sqrt(gamma * n_th_a) * sp);

  vector<double> times(100000);
  for (int i = 0; i < times.size(); ++i) {
    times[i] = i * 250.0 / (times.size() - 1);
  }

  auto start_time1 = high_resolution_clock::now();
  auto result_ode = montecarlo(H, c_op_list, psi0, times, 42, true);
  auto end_time1 = high_resolution_clock::now();
  cout << "Time taken using GSL: "
       << duration_cast<milliseconds>(end_time1 - start_time1).count()
       << " milliseconds" << endl;

  auto start_time2 = high_resolution_clock::now();
  auto result_exp = montecarlo(H, c_op_list, psi0, times, 42, false);
  auto end_time2 = high_resolution_clock::now();
  cout << "Time taken using matrix exponentiation: "
       << duration_cast<milliseconds>(end_time2 - start_time2).count()
       << " milliseconds" << endl;

  vector<double> norms_ode;
  for (const auto& psi : result_ode) {
    norms_ode.push_back(psi.squaredNorm());
  }
  vector<double> norms_exp;
  for (const auto& psi : result_exp) {
    norms_exp.push_back(psi.squaredNorm());
  }

  using namespace matplot;

  plot(times, norms_ode, "-r", times, norms_exp, "-b");
  xlabel("Time");
  ylabel("Norm of psi");
  legend({"GSL", "Matrix Exponentiation"});
  show();

  // Plotting and other operations can be done similarly using appropriate C++
  // libraries

  return 0;
}
