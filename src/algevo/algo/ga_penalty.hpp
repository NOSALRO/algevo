#ifndef ALGEVO_ALGO_GA_PENALTY_HPP
#define ALGEVO_ALGO_GA_PENALTY_HPP

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
class GeneticAlgorithmPenalty {
public:
    using population_t = Eigen::Matrix<Scalar, -1, -1>;
    using x_t = Eigen::Matrix<Scalar, -1, 1>;
    using fit_eval_t = std::vector<Fit>;
    using idx_t = std::vector<unsigned int>;

    using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
    using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
    using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
    using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;

    struct Params {
        int seed = -1;

        unsigned int dim = 0;
        unsigned int pop_size = 0;
        unsigned int num_elites = 0;

        Scalar sigma_1 = static_cast<Scalar>(0.01);
        Scalar sigma_2 = static_cast<Scalar>(0.2);

        unsigned int neq_dim = 0;
        unsigned int nin_dim = 0;

        x_t min_value;
        x_t max_value;

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

    GeneticAlgorithmPenalty(const Params& params, const Fit& init_fit = {})
        : _params(params), _num_elites(std::min(params.num_elites, params.pop_size / 2)), _rgen(0., 1., params.seed)
    {
        assert(_params.pop_size >= 2 && "Population size needs to be bigger than 1!");
        assert(_params.dim > 0 && "Dimensions not set!");
        assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");

        _allocate_data(init_fit);

        for (unsigned int i = 0; i < _params.pop_size; i++) {
            for (unsigned int j = 0; j < _params.dim; j++) {
                Scalar range = (_params.max_value[j] - _params.min_value[j]);
                _population(j, i) = _rgen.rand() * range + _params.min_value[j];
            }
        }

        _fit_best = -std::numeric_limits<Scalar>::max();
        _cv_best = std::numeric_limits<Scalar>::max();
    }

    IterationLog step(bool force_reeval = false)
    {
        _evaluate_population(force_reeval);
        _sort_population();
        _genetic_operators();

        _log.iterations++;
        if (force_reeval || _log.iterations == 1)
            _log.func_evals += _params.pop_size;
        else
            _log.func_evals += _num_elites;

        _log.best = _best;
        _log.best_value = _fit_best;
        _log.best_cv = _cv_best;

        return _log;
    }

    const population_t& population() const { return _population; }
    population_t& population() { return _population; }

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

    x_t _best;
    Scalar _fit_best;
    Scalar _cv_best;

    idx_t _best_idxs;
    unsigned int _num_elites;

    fit_eval_t _fit_evals;
    rgen_scalar_t _rgen;
    rgen_scalar_gauss_t _rgen_gauss = rgen_scalar_gauss_t(static_cast<Scalar>(0.), static_cast<Scalar>(1.));

    struct EvalData {
        Scalar value = 0.;
        Scalar constraint_violation = 0.;
        x_t constraints;
    };

    std::vector<EvalData> _eval_data;

    void _allocate_data(const Fit& init_fit = {})
    {
        _population = population_t(_params.dim, _params.pop_size);
        _population_fit = x_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());
        _population_cv = x_t::Constant(_params.pop_size, std::numeric_limits<Scalar>::max());
        _best = x_t::Constant(_params.dim, 0.);

        _fit_evals.resize(_params.pop_size, init_fit);
        _best_idxs.resize(_params.pop_size);
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

    Scalar _penalty_weight() const
    {
        Scalar h = static_cast<Scalar>(_log.iterations + 1);
        return h * std::sqrt(h);
    }

    void _evaluate_population(bool force_reeval = false)
    {
        if (_log.iterations == 0 || force_reeval) {
            tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                auto res = _fit_evals[i].eval_all(_population.col(i));

                _eval_data[i].value = std::get<0>(res);
                _eval_data[i].constraints = std::get<1>(res);
                _eval_data[i].constraint_violation = std::get<2>(res);

                _population_fit(i) = _eval_data[i].value;
                _population_cv(i) = _compute_penalty(_eval_data[i]);
            });
        } else {
            tools::parallel_loop(0, _num_elites, [this](size_t i) {
                unsigned int idx = _best_idxs[_params.pop_size - 1 - i];
                auto res = _fit_evals[idx].eval_all(_population.col(idx));

                _eval_data[idx].value = std::get<0>(res);
                _eval_data[idx].constraints = std::get<1>(res);
                _eval_data[idx].constraint_violation = std::get<2>(res);

                _population_fit(idx) = _eval_data[idx].value;
                _population_cv(idx) = _compute_penalty(_eval_data[idx]);
            });
        }
    }

    void _sort_population()
    {
        std::iota(_best_idxs.begin(), _best_idxs.end(), 0);

        Scalar h = _penalty_weight();
        std::sort(_best_idxs.begin(), _best_idxs.end(),
                  [this, h](size_t i1, size_t i2) {
                      return _compare(_population_fit(i1), _population_cv(i1),
                                      _population_fit(i2), _population_cv(i2), h);
                  });

        _fit_best = _population_fit[_best_idxs[0]];
        _cv_best = _population_cv[_best_idxs[0]];
        _best = _population.col(_best_idxs[0]);
    }

    void _genetic_operators()
    {
        tools::parallel_loop(0, _num_elites, [this](size_t i) {
            _mutation_and_crossover(i);
        });
    }

    void _mutation_and_crossover(unsigned int elite_idx)
    {
        static thread_local tools::rgen_int_t rgen_elites(0, _num_elites - 1, _params.seed);

        unsigned int p1 = _best_idxs[elite_idx];
        unsigned int p2 = _best_idxs[rgen_elites.rand()];

        while (p1 == p2) {
            p2 = _best_idxs[rgen_elites.rand()];
        }

        _population.col(_best_idxs[_params.pop_size - elite_idx - 1]) =
            _population.col(p1) + _params.sigma_2 * _rgen_gauss.rand() * (_population.col(p2) - _population.col(p1));

        for (unsigned int j = 0; j < _params.dim; j++) {
            _population(j, _best_idxs[_params.pop_size - elite_idx - 1]) += _rgen_gauss.rand() * _params.sigma_1;
            _population(j, _best_idxs[_params.pop_size - elite_idx - 1]) =
                std::max(_params.min_value[j],
                         std::min(_params.max_value[j], _population(j, _best_idxs[_params.pop_size - elite_idx - 1])));
        }
    }
};

} // namespace algo
} // namespace algevo

#endif