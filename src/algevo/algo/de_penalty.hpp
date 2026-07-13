#ifndef ALGEVO_ALGO_DE_PENALTY_HPP
#define ALGEVO_ALGO_DE_PENALTY_HPP

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
namespace algo {

template <typename Fit, typename Scalar = double>
class DifferentialEvolutionPenalty {
public:
    using population_t = Eigen::Matrix<Scalar, -1, -1>;
    using x_t = Eigen::Matrix<Scalar, -1, 1>;
    using fit_eval_t = std::vector<Fit>;

    using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
    using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
    using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
    using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;

    struct Params {
        int seed = -1;
        Scalar cr = static_cast<Scalar>(0.9);
        Scalar f = static_cast<Scalar>(0.8);
        Scalar lambda = static_cast<Scalar>(0.8);

        unsigned int dim = 0;
        unsigned int pop_size = 0;
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

    DifferentialEvolutionPenalty(const Params& params, const Fit& init_fit = {})
        : _params(params), _rgen(0., 1., params.seed)
    {
        assert(_params.pop_size >= 3 && "Population size needs to be bigger than 2!");
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

    IterationLog step()
    {
        tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
            _update_candidate(i);
        });

        for (unsigned int i = 0; i < _params.pop_size; i++) {
            if (_compare(_population_fit(i), _population_cv(i), _fit_best, _cv_best, _penalty_weight())) {
                _fit_best = _population_fit(i);
                _cv_best = _population_cv(i);
                _best = _population.col(i);
            }
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

    fit_eval_t _fit_evals;
    rgen_scalar_t _rgen;

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

    void _update_candidate(unsigned int i)
    {
        static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);
        static thread_local tools::rgen_int_t rgen_dim(0, _params.dim - 1, _params.seed);
        static thread_local tools::rgen_int_t rgen_pop(0, _params.pop_size - 1, _params.seed);

        Scalar cr = _params.cr;
        Scalar f = _params.f;
        Scalar l = _params.lambda;

        unsigned int i1 = rgen_pop.rand(), i2 = rgen_pop.rand();
        while (i1 == i) {
            i1 = rgen_pop.rand();
        }
        while (i2 == i || i2 == i1) {
            i2 = rgen_pop.rand();
        }

        unsigned int R = rgen_dim.rand();

        x_t y = _population.col(i);
        for (unsigned int j = 0; j < _params.dim; j++) {
            if (j == R || rgen.rand() < cr) {
                Scalar v = 0.;
                if (_log.iterations > 0)
                    v = l * (_best(j) - _population(j, i));
                y(j) = std::min(_params.max_value[j], std::max(_params.min_value[j], _population(j, i) + v + f * (_population(j, i1) - _population(j, i2))));
            }
        }

        auto res = _fit_evals[i].eval_all(y);

        _eval_data[i].value = std::get<0>(res);
        _eval_data[i].constraints = std::get<1>(res);
        _eval_data[i].constraint_violation = std::get<2>(res);

        Scalar perf = _eval_data[i].value;
        Scalar pen = _compute_penalty(_eval_data[i]);

        if (_log.iterations == 0 || _compare(perf, pen, _population_fit(i), _population_cv(i), _penalty_weight())) {
            _population_fit(i) = perf;
            _population_cv(i) = pen;
            _population.col(i) = y;
        }
    }
};

} // namespace algo
} // namespace algevo

#endif