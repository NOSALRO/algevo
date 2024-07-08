#include <iostream>

#include <algevo/algo/cem_md.hpp>

template <typename Scalar = double> struct FitMixed {
  static constexpr unsigned int dim_discrete = 4;
  static constexpr unsigned int dim_continuous = 4;
  static constexpr unsigned int num_values = 2;

  using xd_t = Eigen::Matrix<unsigned int, 1, dim_discrete>;
  using xc_t = Eigen::Matrix<double, 1, dim_continuous>;

  Scalar eval(const xd_t &xd, const xc_t &xc) {
    double sum = 0.;
    for (size_t i = 0; i < dim_continuous; ++i) {
      if (xd[i] == 1)
        sum += xc[i];
    }

    return -std::pow(sum - 1., 2);
  }
};

// Typedefs
using FitD = FitMixed<>;
using Algo = algevo::algo::CrossEntropyMethodMixed<FitD>;
using Params = Algo::Params;

int main() {
  // Set parameters
  Params params;
  params.dim_discrete = FitD::dim_discrete;
  params.dim_continuous = FitD::dim_continuous;
  params.pop_size = 24;
  params.num_elites = params.pop_size * 0.8;
  params.num_values = {FitD::num_values, FitD::num_values, FitD::num_values,
                       FitD::num_values};
  params.init_probs = Algo::p_t::Ones(FitD::dim_discrete, FitD::num_values) /
                      static_cast<double>(FitD::num_values);

  params.max_value_continuous = Algo::xc_t::Constant(params.dim_continuous, 2.);
  params.min_value_continuous =
      Algo::xc_t::Constant(params.dim_continuous, -2.);
  params.init_mu_continuous = Algo::xc_t::Constant(params.dim_continuous, 1.);
  params.init_std_continuous = Algo::xc_t::Constant(params.dim_continuous, 1.);

  // Instantiate algorithm
  Algo cem(params);

  // Run a few iterations!
  for (unsigned int i = 0; i < 1000; i++) {
    auto log = cem.step();
    std::cout << log.iterations << "(" << log.func_evals
              << "): " << log.best_value << std::endl;
    // std::cout << log.best.transpose() << std::endl;
    std::cout << "Discrete Probabilities: \n"
              << cem.probabilities() << std::endl;
    std::cout << "Mean: \n" << cem.mu() << std::endl;
    std::cout << "Sigma: \n" << cem.std_devs() << std::endl << std::endl;
  }

  return 0;
}
