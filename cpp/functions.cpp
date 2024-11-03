#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include <Eigen/Dense>
#include <boost/random.hpp>
#include <complex>
#include <iostream>
#include <random>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

using namespace Eigen;
using namespace std;

typedef Matrix<complex<double>, Dynamic, Dynamic> MatrixXcd;
typedef Matrix<complex<double>, Dynamic, 1> VectorXcd;

double prand(boost::random::mt19937 &gen) {
  int a = gen() >> 5;
  int b = gen() >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

struct Params {
  MatrixXcd H;
};

int schrodinger_eq(double t, const double y[], double f[], void *params) {
  Params *p = (Params *)params;
  VectorXcd psi(p->H.rows());
  for (int i = 0; i < p->H.rows(); ++i) {
    psi[i] = complex<double>(y[2 * i], y[2 * i + 1]);
  }
  Map<VectorXcd> dpsi_dt(reinterpret_cast<complex<double> *>(f), p->H.rows());
  dpsi_dt = -1i * p->H * psi;
  return GSL_SUCCESS;
}

VectorXcd integrate(const MatrixXcd &H, const VectorXcd &psi0, double t_span,
                    double t_eval, bool use_ode = true) {
  VectorXcd psi = psi0;
  if (use_ode) {
    gsl_odeiv2_system sys = {schrodinger_eq, nullptr, H.rows() * 2,
                             new Params{H}};
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_msadams, 1e-8, 1e-8, 1e-6);
    double y[psi.size() * 2];
    for (int i = 0; i < psi.size(); ++i) {
      y[2 * i] = real(psi[i]);
      y[2 * i + 1] = imag(psi[i]);
    }
    gsl_odeiv2_driver_apply(d, &t_span, t_eval, y);
    for (int i = 0; i < psi.size(); ++i) {
      psi[i] = complex<double>(y[2 * i], y[2 * i + 1]);
    }
    gsl_odeiv2_driver_free(d);
    delete (Params *)sys.params;
  } else {
    auto dt = t_eval - t_span;
    MatrixXcd U = (-1i * H * dt).exp();
    psi = U * psi;
  }
  return psi;
}

MatrixXcd kroneckerProduct(const MatrixXcd &A, const MatrixXcd &B) {
  MatrixXcd result(A.rows() * B.rows(), A.cols() * B.cols());
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j) {
      result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) =
          A(i, j) * B;
    }
  }
  return result;
}

VectorXcd vkroneckerProduct(const VectorXcd &A, const VectorXcd &B) {
  VectorXcd result(A.size() * B.size());
  for (int i = 0; i < A.size(); ++i) {
    for (int j = 0; j < B.size(); ++j) {
      result[i * B.size() + j] = A[i] * B[j];
    }
  }
  return result;
}

MatrixXcd determine_jump(const vector<MatrixXcd> &c_ops, const VectorXcd &psi,
                         double r2) {
  MatrixXcd jump;
  double p_tot = 0;
  for (const auto &c_op : c_ops) {
    p_tot += (c_op * psi).squaredNorm();
  }
  double p_it = 0;
  for (const auto &c_op : c_ops) {
    double prob = (c_op * psi).squaredNorm();
    p_it += prob / p_tot;
    if (p_it >= r2) {
      jump = c_op;
      break;
    }
  }
  return jump;
}

vector<VectorXcd> montecarlo(const MatrixXcd &H, const vector<MatrixXcd> &c_ops,
                             const VectorXcd &psi0, const vector<double> &tlist,
                             unsigned int seed = 0, bool use_ode = true) {
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  boost::random::mt19937 gen(seed);

  MatrixXcd H_eff = H;
  for (const auto &c_op : c_ops) {
    H_eff -= 0.5i * c_op.adjoint() * c_op;
  }

  VectorXcd psi = psi0 / psi0.norm();
  vector<VectorXcd> psi_j;
  psi_j.reserve(tlist.size());
  psi_j.push_back(psi);
  double t_prev = tlist[0];
  double r1 = prand(gen);
  double r2 = prand(gen);

  for (size_t t_idx = 1; t_idx < tlist.size(); ++t_idx) {
    double t_span = t_prev;
    double t_eval = tlist[t_idx];

    psi = integrate(H_eff, psi, t_span, t_eval, use_ode);

    double norm_sq = psi.squaredNorm();
    if (norm_sq <= r1) {
      MatrixXcd jump = determine_jump(c_ops, psi, r2);
      psi = jump * psi / (jump * psi).norm();
      r1 = prand(gen);
      r2 = prand(gen);
    }

    t_prev = t_eval;
    psi_j.push_back(psi);
  }

  return psi_j;
}

vector<complex<double>> montecarlo_average(
    const MatrixXcd &H, const vector<MatrixXcd> &c_ops, const VectorXcd &psi0,
    const vector<double> &tlist, size_t const ntraj, const MatrixXcd &op,
    unsigned int seed = 0, bool use_ode = true) {
  auto t_size{tlist.size()};
  vector<complex<double>> averages(t_size, 0);
  for (int i = 0; i < ntraj; ++i) {
    auto result{montecarlo(H, c_ops, psi0, tlist, seed + i, use_ode)};
    for (size_t i = 0; i < t_size; ++i) {
      averages[i] +=
          (result[i].adjoint() * op * result[i])(0, 0) / result[i].squaredNorm();
    }
  }

  for (auto &avg : averages) {
    avg /= ntraj;
  }
  return averages;
}
