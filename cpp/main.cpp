#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <matplot/matplot.h>

#include <Eigen/Dense>
#include <boost/program_options.hpp>
#include <chrono>
#include <complex>
#include <iostream>
#include <numbers>
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

int main(int argc, char* argv[]) {
  // Define variables with default values
  int N = 2;
  double wc = 1.0e-3 * 2 * std::numbers::pi;
  double wa = wc;
  double g = 0.05 * 2 * std::numbers::pi;
  double kappa = 0.005;
  double gamma = 0.05;
  double n_th_a = 0.1;
  int n_times = 1000;
  int n_traj = 500;
  double time_f = 50.0;
  int seed = 0;

  // Set up Boost Program Options
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "N", po::value<int>(&N)->default_value(N), "value of N")(
      "wc", po::value<double>(&wc)->default_value(wc), "value of wc")(
      "wa", po::value<double>(&wa)->default_value(wa), "value of wa")(
      "g", po::value<double>(&g)->default_value(g), "value of g")(
      "kappa", po::value<double>(&kappa)->default_value(kappa),
      "value of kappa")("gamma",
                        po::value<double>(&gamma)->default_value(gamma),
                        "value of gamma")(
      "n_th_a", po::value<double>(&n_th_a)->default_value(n_th_a),
      "value of n_th_a")("n_times",
                         po::value<int>(&n_times)->default_value(n_times),
                         "number of times")(
      "n_traj", po::value<int>(&n_traj)->default_value(n_traj),
      "number of trajectories")(
      "time_f", po::value<double>(&time_f)->default_value(time_f),
      "final time")("seed", po::value<int>(&seed)->default_value(seed),
                    "random seed");

  // Parse command line arguments
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Display help message if needed
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  VectorXcd psi0_c = VectorXcd::Zero(N, 1);
  psi0_c(0) = std::complex<double>(1.0, 0.0);
  VectorXcd psi0_a = VectorXcd::Zero(2, 1);
  psi0_a(1) = std::complex<double>(1.0, 0.0);
  VectorXcd psi0 = kroneckerProduct(psi0_c, psi0_a).eval();

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

  auto e_op = sp * sm;

  vector<double> times(n_times, 0);
  for (int i = 0; i < n_times; ++i) {
    times[i] = i * time_f / (times.size() - 1);
  }

  auto start_time1 = high_resolution_clock::now();
  // auto result_ode = montecarlo(H, c_op_list, psi0, times, 42, true);
  auto result_ode =
      montecarlo_average(H, c_op_list, psi0, times, n_traj, e_op, seed, true);
  auto end_time1 = high_resolution_clock::now();
  cout << "Time taken using GSL: "
       << duration_cast<milliseconds>(end_time1 - start_time1).count()
       << " milliseconds" << endl;

  auto start_time2 = high_resolution_clock::now();
  // auto result_exp = montecarlo(H, c_op_list, psi0, times, 42, false);
  auto result_exp =
      montecarlo_average(H, c_op_list, psi0, times, n_traj, e_op, seed, false);
  auto end_time2 = high_resolution_clock::now();
  cout << "Time taken using matrix exponentiation: "
       << duration_cast<milliseconds>(end_time2 - start_time2).count()
       << " milliseconds" << endl;

  /*
    vector<double> norms_ode;
    for (const auto& psi : result_ode) {
      norms_ode.push_back(psi.squaredNorm());
    }
    vector<double> norms_exp;
    for (const auto& psi : result_exp) {
      norms_exp.push_back(psi.squaredNorm());
    }
  */

  using namespace matplot;

  vector<double> result_ode_real(result_ode.size());
  for (size_t i = 0; i < result_ode.size(); ++i) {
    result_ode_real[i] = result_ode[i].real();
  }
  vector<double> result_exp_real(result_exp.size());
  for (size_t i = 0; i < result_exp.size(); ++i) {
    result_exp_real[i] = result_exp[i].real();
  }
  plot(times, result_ode_real, "-r", times, result_exp_real, "-b");
  xlabel("Time");
  ylabel("P_e(t)");
  legend({"GSL", "Matrix Exponentiation"});
  show();

  // Plotting and other operations can be done similarly using appropriate C++
  // libraries

  return 0;
}
