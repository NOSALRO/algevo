#ifndef ALGEVO_ALGO_CEM_MIXED_HPP
#define ALGEVO_ALGO_CEM_MIXED_HPP

#include <eigen3/Eigen/Core>

#include <algorithm> // std::sort, std::stable_sort
#include <limits>
#include <numeric> // std::iota

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
namespace algo {
template <typename Fit, typename Scalar = double>
class CrossEntropyMethodMixed {
public:
  using population_discrete_t = Eigen::Matrix<unsigned int, -1, -1>;
  using population_continuous_t = Eigen::Matrix<Scalar, -1, -1>;
  using xd_t = Eigen::Matrix<unsigned int, -1, 1>;
  using xc_t = Eigen::Matrix<Scalar, -1, 1>;
  using p_t = Eigen::Matrix<Scalar, -1, -1>;
  using fit_t = Eigen::Matrix<Scalar, 1, -1>;
  using fit_eval_t = std::vector<Fit>;

  using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
  using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
  using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
  using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;
  using colored_noise_t = tools::ColoredNoiseGenerator<Scalar>;

  struct Params {
    int seed = -1;

    // unsigned int dim = 0;
    unsigned int pop_size = 0;
    unsigned int num_elites = 0;

    // Discrete
    unsigned int dim_discrete = 0;
    std::vector<unsigned int> num_values;
    p_t init_probs;

    // Continuous
    unsigned int dim_continuous = 0;
    xc_t min_value_continuous;
    xc_t max_value_continuous;

    xc_t init_mu_continuous;
    xc_t init_std_continuous;

    xc_t min_std_continuous;

    Scalar decrease_pop_factor = 1.;
    Scalar fraction_elites_reused = 0.;

    Scalar prob_keep_previous =
        0.; // < 0, colored noise, ==0. regular noise, > 0. sticky actions
    Scalar beta = 1.; // exponent for colored noise
    unsigned int elem_size = 0;
  };

  struct IterationLog {
    unsigned int iterations = 0;
    unsigned int func_evals = 0;

    xd_t best_discrete;
    xc_t best_continuous;
    Scalar best_value;
  };

  CrossEntropyMethodMixed(const Params &params)
      : _params(params), _update_coeff(static_cast<Scalar>(1.) /
                                       static_cast<Scalar>(_params.num_elites)),
        _elites_reuse_size(
            std::max(0u, std::min(_params.num_elites,
                                  static_cast<unsigned int>(
                                      _params.num_elites *
                                      _params.fraction_elites_reused)))),
        _rgen(0., 1., params.seed), _colored_rgen(params.seed) {
    assert(_params.pop_size > 0 &&
           "Population size needs to be bigger than zero!");
    assert(_params.dim > 0 && "Dimensions not set!");
    assert(_params.num_elites > 0 && _params.num_elites <= _params.pop_size &&
           "Number of elites is wrongly set!");

    _allocate_data_discrete();
    _allocate_data_continuous();

    _population_fit =
        xc_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());

    _fit_evals.resize(_params.pop_size);

    ////////////////////////////////////////
    _fit_best = -std::numeric_limits<Scalar>::max();
  }

  IterationLog step() {
    // Generate population
    _generate_population_discrete();
    _generate_population_continuous();

    // Evaluate population
    _evaluate_population();

    // Update probabilities
    _update_distribution_discrete();
    _update_distribution_continuous();

    // Update iteration log
    _log.iterations++;
    _log.func_evals += _params.pop_size;
    _log.best_discrete = _best_discrete;
    _log.best_continuous = _best_continuous;
    _log.best_value = _fit_best;

    return _log;
  }

  const population_discrete_t &population_discrete() const {
    return _population_discrete;
  }
  population_discrete_t &population_discrete() { return _population_discrete; }

  const population_continuous_t &population_continuous() const {
    return _population_continuous;
  }
  population_continuous_t &population_continuous() {
    return _population_continuous;
  }

  const p_t &probabilities() const { return _probs; }

  const xc_t &mu() const { return _mu; }
  const xc_t &std_devs() const { return _std_devs; }

  const xc_t &population_fit() const { return _population_fit; }

  // const x_t &best() const { return _best; }
  Scalar best_value() const { return _fit_best; }

protected:
  // Parameters
  Params _params;

  // Iteration Log
  IterationLog _log;

  // Actual population (current values)
  population_discrete_t _population_discrete;
  population_continuous_t _population_continuous;
  xc_t _population_fit;

  // Categorical Distribution
  p_t _probs;

  // Gaussian Distribution
  xc_t _mu;
  xc_t _std_devs;

  // Data for updates
  const Scalar _update_coeff;
  population_discrete_t _elites_discrete;
  population_continuous_t _elites_continuous;

  const unsigned int _elites_reuse_size;

  // Best ever
  xd_t _best_discrete;
  xc_t _best_continuous;
  Scalar _fit_best;

  // Evaluators
  fit_eval_t _fit_evals;

  // Random numbers
  rgen_scalar_gauss_t _rgen;
  colored_noise_t _colored_rgen;

  void _allocate_data_discrete() {
    _population_discrete =
        population_discrete_t(_params.dim_discrete, _params.pop_size);
    _elites_discrete =
        population_discrete_t(_params.dim_discrete, _params.num_elites);

    _best_discrete = xd_t::Constant(_params.dim_discrete, 0);

    _probs = _params.init_probs;
  }

  void _allocate_data_continuous() {
    _population_continuous =
        population_continuous_t(_params.dim_continuous, _params.pop_size);
    _elites_continuous =
        population_continuous_t(_params.dim_continuous, _params.num_elites);

    _best_continuous = xc_t::Constant(_params.dim_continuous,
                                      -std::numeric_limits<Scalar>::max());

    _mu = _params.init_mu_continuous;
    _std_devs = _params.init_std_continuous;
  }

  void _generate_population_discrete() {
    static thread_local rgen_scalar_t rgen(
        static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);

    // Generate random gaussian values from pure Normal distribution (mean=0,
    // std=1)
    for (unsigned int i = 0; i < _params.pop_size; i++) {
      for (unsigned int j = 0; j < _params.dim_discrete; j++) {
        Scalar p = rgen.rand();
        Scalar s = static_cast<Scalar>(0.);
        unsigned int k = 0;
        for (; k < _params.num_values[j]; k++) {
          s += _probs(j, k);
          if (p < s)
            break;
        }
        _population_discrete(j, i) = k;
      }
    }
  }

  void _generate_population_continuous(bool inject_mean_to_population = false) {
    static thread_local rgen_scalar_t rgen(
        static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);

    // classic generation of population
    // Generate random gaussian values from pure Normal distribution (mean=0,
    // std=1)
    for (unsigned int i = 0; i < _params.pop_size; i++) {
      for (unsigned int j = 0; j < _params.dim_continuous; j++) {
        _population_continuous(j, i) = _rgen.rand();
      }
    }

    // Convert random values (sampled from N(0,1)) into a population: mu_{i+1} =
    // mu_i + sigma_i * random_eps
    _population_continuous =
        (_population_continuous.array().colwise() * _std_devs.array())
            .colwise() +
        _mu.array();

    // Clamp inside min/max
    for (unsigned int i = 0; i < _params.pop_size; i++) {
      for (unsigned int j = 0; j < _params.dim_continuous; j++) {
        _population_continuous(j, i) =
            std::max(_params.min_value_continuous[j],
                     std::min(_params.max_value_continuous[j],
                              _population_continuous(j, i)));
      }
    }

    // Inject elites from previous inner iteration
    if (_log.iterations > 0)
      for (unsigned int i = 0; i < _elites_reuse_size; i++)
        _population_continuous.col(i) = _elites_continuous.col(i);
    if (inject_mean_to_population)
      _population_continuous.col(_elites_reuse_size) = _mu;
  }

  void _evaluate_population() {
    // Evaluate individuals
    tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
      _population_fit[i] = _fit_evals[i].eval(_population_discrete.col(i),
                                              _population_continuous.col(i));
    });

    // Update global best
    for (unsigned int i = 0; i < _params.pop_size; i++) {
      if (_population_fit[i] > _fit_best) {
        _fit_best = _population_fit[i];
        _best_discrete =
            _population_discrete.col(i); // TO-DO: Maybe tag to avoid copies?
        _best_continuous =
            _population_continuous.col(i); // TO-DO: Maybe tag to avoid copies?
      }
    }
  }

  void _update_distribution_discrete() {
    // Sort individuals by their perfomance (best first!)
    std::vector<unsigned int> idx(_params.pop_size);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [this](unsigned int i1, unsigned int i2) {
      return _population_fit[i1] > _population_fit[i2];
    });

    for (unsigned int i = 0; i < _params.num_elites; i++)
      _elites_discrete.col(i) = _population_discrete.col(idx[i]);

    // Update probabilities using the elites!
    for (unsigned int j = 0; j < _params.dim_discrete; j++) {
      std::vector<unsigned int> counter(_params.num_values[j], 0);
      for (unsigned int i = 0; i < _params.num_elites; i++) {
        counter[_elites_discrete(j, i)]++;
      }
      for (unsigned int k = 0; k < _params.num_values[j]; k++) {
        _probs(j, k) = static_cast<Scalar>(counter[k]) /
                       static_cast<Scalar>(_params.num_elites);
      }
    }
  }

  void _update_distribution_continuous() {
    // Sort individuals by their perfomance (best first!)
    std::vector<unsigned int> idx(_params.pop_size);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [this](unsigned int i1, unsigned int i2) {
      return _population_fit[i1] > _population_fit[i2];
    });

    for (unsigned int i = 0; i < _params.num_elites; i++)
      _elites_continuous.col(i) = _population_continuous.col(idx[i]);

    // Update mean/variance using the elites!
    _std_devs =
        (_update_coeff * (_elites_continuous.array().colwise() - _mu.array())
                             .square()
                             .rowwise()
                             .sum())
            .sqrt(); // sample variance
    if (_params.min_std_continuous.size() == _std_devs.size()) {
      for (unsigned int i = 0; i < _params.dim_continuous; i++)
        _std_devs(i) = std::max(_params.min_std_continuous(i), _std_devs(i));
    }
    _mu = _update_coeff * _elites_continuous.rowwise().sum(); // sample mean
  }
};
} // namespace algo
} // namespace algevo

#endif
