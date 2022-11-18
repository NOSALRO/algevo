#ifndef ALGEVO_ALGO_DE_HPP
#define ALGEVO_ALGO_DE_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        template <typename Params, typename Fit, typename Scalar = double>
        class DifferentialEvolution {
        public:
            // This is allocating too much stack memory, also no actual performance gain
            // using population_t = Eigen::Matrix<Scalar, Params::pop_size, Params::dim>;
            // using fit_t = Eigen::Matrix<Scalar, Params::pop_size, 1>;
            // using neighborhoods_t = Eigen::Matrix<Scalar, Params::num_neighborhoods, Params::dim>;
            // using neighborhood_fit_t = Eigen::Matrix<Scalar, Params::num_neighborhoods, 1>;
            // using x_t = Eigen::Matrix<Scalar, 1, Params::dim>;
            using population_t = Eigen::Matrix<Scalar, -1, -1>;
            using fit_t = Eigen::Matrix<Scalar, -1, 1>;
            using neighborhoods_t = Eigen::Matrix<Scalar, -1, -1>;
            using neighborhood_fit_t = Eigen::Matrix<Scalar, -1, 1>;
            using x_t = Eigen::Matrix<Scalar, 1, -1>;

            using fit_eval_t = std::array<Fit, Params::pop_size>;

            using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
            using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
            using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
            using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;

            DifferentialEvolution()
            {
                static_assert(Params::pop_size >= 4, "Population size needs to be bigger than 3!");
                _allocate_data();

                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    for (unsigned int j = 0; j < Params::dim; j++) {
                        _population(i, j) = _rgen.rand();
                    }
                }

                _fit_best = -std::numeric_limits<Scalar>::max();

                // Evaluate initial population population
                _evaluate_population();

                // Update global best
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    if (_population_fit(i) > _fit_best) {
                        _fit_best = _population_fit(i);
                        _best = _population.row(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }
            }

            void step()
            {
                // Do updates (evaluates new candidates as well)
                tools::parallel_loop(0, Params::pop_size, [this](size_t i) {
                    _update_candidate(i);
                });

                // Update global best
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    if (_population_fit(i) > _fit_best) {
                        _fit_best = _population_fit(i);
                        _best = _population.row(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

            const fit_t& population_fit() const { return _population_fit; }

            const x_t& best() const { return _best; }
            Scalar best_value() const { return _fit_best; }

        protected:
            // Actual population (current values)
            population_t _population;

            // Latest fitness evaluations
            fit_t _population_fit;

            // Best ever
            x_t _best;
            Scalar _fit_best;

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_t _rgen = rgen_scalar_t(Params::min_value, Params::max_value, Params::seed);

            void _allocate_data()
            {
                _population = population_t(Params::pop_size, Params::dim);
                _population_fit = fit_t(Params::pop_size);
                _best = x_t(Params::dim);
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, Params::pop_size, [this](size_t i) {
                    _population_fit(i) = _fit_evals[i].eval(_population.row(i));
                });
            }

            void _update_candidate(unsigned int i)
            {
                static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), Params::seed);
                static thread_local tools::rgen_int_t rgen_dim(0, Params::dim - 1, Params::seed);
                static thread_local tools::rgen_int_t rgen_pop(0, Params::pop_size - 1, Params::seed);

                static constexpr Scalar cr = Params::cr;
                static constexpr Scalar f = Params::f;

                // DE/rand-to best/1
                unsigned int i1 = rgen_pop.rand(), i2 = rgen_pop.rand(); //, i3 = rgen_pop.rand();

                // Sample 2 distinct candidates
                while (i1 == i) {
                    i1 = rgen_pop.rand();
                }

                while (i2 == i || i2 == i1) {
                    i2 = rgen_pop.rand();
                }

                // while (i3 == i || i3 == i1 || i3 == i2) {
                //     i3 = rgen_pop.rand();
                // }

                unsigned int R = rgen_dim.rand();

                x_t y = _population.row(i); // copy original candidate
                for (unsigned int j = 0; j < Params::dim; j++) {
                    if (j == R || rgen.rand() < cr) {
                        // TO-DO: Maybe mutex is needed?
                        // y(j) = _population(i1, j) + f * (_population(i2, j) - _population(i3, j));
                        y(j) = std::min(Params::max_value, std::max(Params::min_value, _population(i, j) + f * (_best(j) - _population(i, j)) + f * (_population(i1, j) - _population(i2, j))));
                    }
                }

                Scalar perf = _fit_evals[i].eval(y);
                if (perf >= _population_fit(i)) {
                    _population_fit(i) = perf;
                    _population.row(i) = y;
                }
            }
        };
    } // namespace algo
} // namespace algevo

#endif
