#ifndef ALGEVO_ALGO_CEM_HPP
#define ALGEVO_ALGO_CEM_HPP

#include <Eigen/Core>

#include <algorithm> // std::sort, std::stable_sort
#include <array>
#include <limits>
#include <numeric> // std::iota

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        template <typename Fit, typename Scalar = double>
        class CrossEntropyMethod {
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
                unsigned int elem_size = 0;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;

                x_t best;
                Scalar best_value;
            };

            CrossEntropyMethod(const Params& params) : _params(params), _update_coeff(static_cast<Scalar>(1.) / static_cast<Scalar>(_params.num_elites)), _elites_reuse_size(std::max(0u, std::min(_params.num_elites, static_cast<unsigned int>(_params.num_elites * _params.fraction_elites_reused)))), _rgen(0., 1., params.seed)
            {
                assert(_params.pop_size > 0 && "Population size needs to be bigger than zero!");
                assert(_params.dim > 0 && "Dimensions not set!");
                assert(_params.num_elites > 0 && _params.num_elites <= _params.pop_size && "Number of elites is wrongly set!");
                assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");
                assert(_params.min_vel.size() == _params.dim && _params.max_vel.size() == _params.dim && "Min/max velocity dimensions should be the same as the problem dimensions!");

                _allocate_data();

                _fit_best = -std::numeric_limits<Scalar>::max();
            }

            IterationLog step(bool inject_mean_to_population = false)
            {
                // Generate population
                _generate_population(inject_mean_to_population);

                // Evaluate population
                _evaluate_population();

                // Update mean/var
                _update_distribution();

                // Update params
                if (_params.decrease_pop_factor > 1.) {
                    _params.pop_size = std::max(_params.num_elites * 2, static_cast<unsigned int>(_params.pop_size / _params.decrease_pop_factor));
                }

                // Update iteration log
                _log.iterations++;
                _log.func_evals += _params.pop_size;
                _log.best = _best;
                _log.best_value = _fit_best;

                return _log;
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

            const x_t& mu() const { return _mu; }
            const x_t& std_devs() const { return _std_devs; }

            const x_t& population_fit() const { return _population_fit; }

            const x_t& best() const { return _best; }
            Scalar best_value() const { return _fit_best; }

        protected:
            // Parameters
            Params _params;

            // Iteration Log
            IterationLog _log;

            // Actual population (current values)
            population_t _population;
            x_t _population_fit;

            // Gaussian Distribution
            x_t _mu;
            x_t _std_devs;

            // Data for updates
            const Scalar _update_coeff;
            const unsigned int _elites_reuse_size;
            population_t _elites;

            // Best ever
            x_t _best;
            Scalar _fit_best;

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_gauss_t _rgen;

            void _allocate_data()
            {
                _population = population_t(_params.dim, _params.pop_size);
                _elites = population_t(_params.dim, _params.num_elites);

                _best = x_t::Constant(_params.dim, -std::numeric_limits<Scalar>::max());
                _population_fit = x_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());

                _mu = _params.init_mu;
                _std_devs = _params.init_std;

                _fit_evals.resize(_params.pop_size);
            }

            void _generate_population(bool inject_mean_to_population)
            {
                static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);

                // Generate data with MPC actions in mind (sticky actions!)
                if (_params.prob_keep_previous > 0. && _params.elem_size) {
                    Scalar prob = _params.prob_keep_previous;
                    for (unsigned int i = 0; i < _params.pop_size; i++) {
                        for (unsigned int j = 0; j < _params.dim; j += _params.elem_size) {
                            if (j > 0 && rgen.rand() < prob) { // keep previous
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
                else { // classic generation of population
                    // Generate random gaussian values from pure Normal distribution (mean=0, std=1)
                    for (unsigned int i = 0; i < _params.pop_size; i++) {
                        for (unsigned int j = 0; j < _params.dim; j++) {
                            _population(j, i) = _rgen.rand();
                        }
                    }

                    // Convert them into a population: mu_{i+1} = mu_i + sigma_i * random_eps
                    _population = (_population.array().colwise() * _std_devs.array()).colwise() + _mu.array();
                }

                // Clamp inside min/max
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        _population(j, i) = std::max(_params.min_value[j], std::min(_params.max_value[j], _population(j, i)));
                    }
                }

                // Inject elites form previous run
                if (_log.iterations == 0 && _params.init_elites.rows() == _params.dim)
                    _population.block(0, 0, _params.dim, _params.init_elites.cols()) = _params.init_elites;

                // Inject elites from previous inner iteration
                if (_log.iterations > 0)
                    for (unsigned int i = 0; i < _elites_reuse_size; i++)
                        _population.col(i) = _elites.col(i);
                if (inject_mean_to_population)
                    _population.col(_elites_reuse_size) = _mu;
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    _population_fit[i] = _fit_evals[i].eval(_population.col(i));
                });

                // Update global best
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if (_population_fit[i] > _fit_best) {
                        _fit_best = _population_fit[i];
                        _best = _population.col(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }
            }

            void _update_distribution()
            {
                // Sort individuals by their perfomance (best first!)
                std::vector<unsigned int> idx(_params.pop_size);
                std::iota(idx.begin(), idx.end(), 0);

                std::sort(idx.begin(), idx.end(), [this](unsigned int i1, unsigned int i2) { return _population_fit[i1] > _population_fit[i2]; });

                for (unsigned int i = 0; i < _params.num_elites; i++)
                    _elites.col(i) = _population.col(idx[i]);

                // Update mean/variance using the elites!
                _std_devs = (_update_coeff * (_elites.array().colwise() - _mu.array()).square().rowwise().sum()).sqrt(); // sample variance
                if (_params.min_std.size() == _std_devs.size()) {
                    for (unsigned int i = 0; i < _params.dim; i++)
                        _std_devs(i) = std::max(_params.min_std(i), _std_devs(i));
                }
                _mu = _update_coeff * _elites.rowwise().sum(); // sample mean
            }
        };
    } // namespace algo
} // namespace algevo

#endif
