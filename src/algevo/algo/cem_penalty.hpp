#ifndef ALGEVO_ALGO_CEM_PENALTY_HPP
#define ALGEVO_ALGO_CEM_PENALTY_HPP

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
namespace algo {

template <typename Fit, typename Scalar = double>
class CrossEntropyMethodPenalty {
public:
    using population_t = Eigen::Matrix<Scalar, -1, -1>;
    using x_t = Eigen::Matrix<Scalar, -1, 1>;
    using fit_eval_t = std::vector<Fit>;

    using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
    using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
    using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
    using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;
    using colored_noise_t = tools::ColoredNoiseGenerator<Scalar>;

    struct Params {
        int seed = -1;

        unsigned int dim = 0;
        unsigned int pop_size = 0;
        unsigned int num_elites = 0;

        x_t min_value;
        x_t max_value;

        x_t init_mu;
        x_t init_std;

        x_t min_std;

        population_t init_elites;

        Scalar decrease_pop_factor = 1.;
        Scalar fraction_elites_reused = 0.;

        Scalar prob_keep_previous = 0.;
        Scalar beta = 1.;
        unsigned int elem_size = 0;

        unsigned int neq_dim = 0;
        unsigned int nin_dim = 0;

        std::vector<std::pair<Scalar, Scalar>> cv_levels = {{0.001, 10.}, {0.1, 20.}, {1., 100.}, {-1., 300.}};
        std::vector<std::pair<Scalar, Scalar>> cv_gamma_levels = {{1., 1.}, {-1., 2.}};
    };

    struct IterationLog {
        unsigned int iterations = 0;
        unsigned int func_evals = 0;

        x_t best;
        Scalar best_value;
        Scalar best_cv;
    };

    CrossEntropyMethodPenalty(const Params& params, const Fit& init_fit = {})
        : _params(params),
          _update_coeff(static_cast<Scalar>(1.) / static_cast<Scalar>(_params.num_elites)),
          _elites_reuse_size(std::max(0u, std::min(_params.num_elites, static_cast<unsigned int>(_params.num_elites * _params.fraction_elites_reused)))),
          _rgen(0., 1., params.seed),
          _colored_rgen(params.seed)
    {
        assert(_params.pop_size > 0 && "Population size needs to be bigger than zero!");
        assert(_params.dim > 0 && "Dimensions not set!");
        assert(_params.num_elites > 0 && _params.num_elites <= _params.pop_size && "Number of elites is wrongly set!");
        assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");

        _allocate_data(init_fit);
        _fit_best = -std::numeric_limits<Scalar>::max();
        _cv_best = std::numeric_limits<Scalar>::max();
    }

    IterationLog step(bool inject_mean_to_population = false)
    {
        _generate_population(inject_mean_to_population);
        _evaluate_population();
        _update_distribution();

        if (_params.decrease_pop_factor > 1.) {
            _params.pop_size = std::max(_params.num_elites * 2, static_cast<unsigned int>(_params.pop_size / _params.decrease_pop_factor));
        }

        _log.iterations++;
        _log.func_evals += _params.pop_size;
        _log.best = _best;
        _log.best_value = _fit_best;
        _log.best_cv = _cv_best;

        return _log;
    }

    const population_t& population() const { return _population; }
    population_t& population() { return _population; }

    const x_t& mu() const { return _mu; }
    const x_t& std_devs() const { return _std_devs; }

    const x_t& population_fit() const { return _population_fit; }
    const x_t& population_cv() const { return _population_cv; }

    const x_t& best() const { return _best; }
    Scalar best_value() const { return _fit_best; }
    Scalar best_cv() const { return _cv_best; }

protected:
    Params _params;
    IterationLog _log;

    population_t _population;
    x_t _population_fit;
    x_t _population_cv;

    x_t _mu;
    x_t _std_devs;

    const Scalar _update_coeff;
    const unsigned int _elites_reuse_size;
    population_t _elites;

    x_t _best;
    Scalar _fit_best;
    Scalar _cv_best;

    fit_eval_t _fit_evals;

    rgen_scalar_gauss_t _rgen;
    colored_noise_t _colored_rgen;

    struct EvalData {
        Scalar value = 0.;
        Scalar constraint_violation = 0.;
        x_t constraints;
    };

    std::vector<EvalData> _eval_data;

    void _allocate_data(const Fit& init_fit = {})
    {
        _population = population_t(_params.dim, _params.pop_size);
        _elites = population_t(_params.dim, _params.num_elites);

        _best = x_t::Constant(_params.dim, 0.);
        _population_fit = x_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());
        _population_cv = x_t::Constant(_params.pop_size, std::numeric_limits<Scalar>::max());

        _mu = _params.init_mu;
        _std_devs = _params.init_std;

        _fit_evals.resize(_params.pop_size, init_fit);
        _eval_data.resize(_params.pop_size);

        for (unsigned int i = 0; i < _params.pop_size; i++) {
            _eval_data[i].constraints = x_t::Zero(_params.neq_dim + _params.nin_dim);
        }
    }

    Scalar _gamma(Scalar val) const
    {
        for (const auto& p : _params.cv_gamma_levels) {
            if (p.first > 0 && val < p.first)
                return p.second;
        }
        return _params.cv_gamma_levels.back().second;
    }

    Scalar _theta(Scalar val) const
    {
        for (const auto& p : _params.cv_levels) {
            if (p.first > 0 && val < p.first)
                return p.second;
        }
        return _params.cv_levels.back().second;
    }

    Scalar _compute_penalty(const EvalData& data) const
    {
        Scalar pen = 0.;

        for (unsigned int i = 0; i < _params.neq_dim; i++) {
            Scalar q = std::abs(data.constraints[i]);
            pen += _theta(q) * q * std::pow(q, _gamma(q));
        }

        for (unsigned int i = 0; i < _params.nin_dim; i++) {
            Scalar q = std::abs(std::min(static_cast<Scalar>(0.), data.constraints[_params.neq_dim + i]));
            pen += _theta(q) * q * std::pow(q, _gamma(q));
        }

        return pen;
    }

    bool _compare(Scalar f1, Scalar pen1, Scalar f2, Scalar pen2, Scalar h) const
    {
        Scalar v1 = -f1 + h * pen1;
        Scalar v2 = -f2 + h * pen2;
        return v1 < v2;
    }

    void _generate_population(bool inject_mean_to_population)
    {
        static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);

        if (_params.prob_keep_previous < 0. && _params.elem_size) {
            unsigned int n_samples = _params.dim / _params.elem_size;
            auto rand_output = _colored_rgen.rand(_params.beta, n_samples, _params.elem_size, _params.pop_size);

            for (unsigned int i = 0; i < _params.pop_size; i++) {
                for (unsigned int j = 0; j < n_samples; j++) {
                    unsigned int k = j * _params.elem_size;
                    for (unsigned int l = 0; l < _params.elem_size; l++) {
                        _population(k + l, i) = rand_output[i][l][j];
                    }
                }
            }
        }
        else if (_params.prob_keep_previous > 0. && _params.elem_size) {
            Scalar prob = _params.prob_keep_previous;
            for (unsigned int i = 0; i < _params.pop_size; i++) {
                for (unsigned int j = 0; j < _params.dim; j += _params.elem_size) {
                    if (j > 0 && rgen.rand() < prob) {
                        _population.block(j, i, _params.elem_size, 1) = _population.block(j - _params.elem_size, i, _params.elem_size, 1);
                        prob *= prob;
                    }
                    else {
                        prob = _params.prob_keep_previous;
                        for (unsigned int k = 0; k < _params.elem_size; k++) {
                            _population(j + k, i) = _rgen.rand();
                        }
                    }
                }
            }
        }
        else {
            for (unsigned int i = 0; i < _params.pop_size; i++) {
                for (unsigned int j = 0; j < _params.dim; j++) {
                    _population(j, i) = _rgen.rand();
                }
            }
        }

        _population = (_population.array().colwise() * _std_devs.array()).colwise() + _mu.array();

        for (unsigned int i = 0; i < _params.pop_size; i++) {
            for (unsigned int j = 0; j < _params.dim; j++) {
                _population(j, i) = std::max(_params.min_value[j], std::min(_params.max_value[j], _population(j, i)));
            }
        }

        if (_log.iterations == 0 && _params.init_elites.rows() == _params.dim)
            _population.block(0, 0, _params.dim, _params.init_elites.cols()) = _params.init_elites;

        if (_log.iterations > 0) {
            for (unsigned int i = 0; i < _elites_reuse_size; i++)
                _population.col(i) = _elites.col(i);
        }

        if (inject_mean_to_population)
            _population.col(_elites_reuse_size) = _mu;
    }

    void _evaluate_population()
    {
        tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
            auto res = _fit_evals[i].eval_all(_population.col(i));

            _eval_data[i].value = std::get<0>(res);
            _eval_data[i].constraints = std::get<1>(res);
            _eval_data[i].constraint_violation = std::get<2>(res);

            _population_fit[i] = _eval_data[i].value;
            _population_cv[i] = _compute_penalty(_eval_data[i]);
        });

        Scalar h = _penalty_weight();
        for (unsigned int i = 0; i < _params.pop_size; i++) {
            if (_compare(_population_fit[i], _population_cv[i], _fit_best, _cv_best, h)) {
                _fit_best = _population_fit[i];
                _cv_best = _population_cv[i];
                _best = _population.col(i);
            }
        }
    }

    void _update_distribution()
    {
        std::vector<unsigned int> idx(_params.pop_size);
        std::iota(idx.begin(), idx.end(), 0);

        Scalar h = _penalty_weight();
        std::sort(idx.begin(), idx.end(),
                  [this, h](unsigned int i1, unsigned int i2) {
                      return _compare(_population_fit[i1], _population_cv[i1],
                                      _population_fit[i2], _population_cv[i2], h);
                  });

        for (unsigned int i = 0; i < _params.num_elites; i++)
            _elites.col(i) = _population.col(idx[i]);

        _std_devs = (_update_coeff * (_elites.array().colwise() - _mu.array()).square().rowwise().sum()).sqrt();

        if (_params.min_std.size() == _std_devs.size()) {
            for (unsigned int i = 0; i < _params.dim; i++)
                _std_devs(i) = std::max(_params.min_std(i), _std_devs(i));
        }

        _mu = _update_coeff * _elites.rowwise().sum();
    }

    Scalar _penalty_weight() const
    {
        Scalar h = static_cast<Scalar>(_log.iterations + 1);
        return h * std::sqrt(h);
    }
};

} // namespace algo
} // namespace algevo

#endif