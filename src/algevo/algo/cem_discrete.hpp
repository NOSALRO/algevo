//|
//|    Copyright (c) 2022-2025 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2023-2025 Laboratory of Automation and Robotics, University of Patras, Greece
//|    Copyright (c) 2022-2025 Konstantinos Chatzilygeroudis
//|    Authors:  Konstantinos Chatzilygeroudis
//|    email:    costashatz@gmail.com
//|    website:  https://nosalro.github.io/
//|              https://lar.upatras.gr/
//|              http://cilab.math.upatras.gr/
//|
//|    This file is part of algevo.
//|
//|    All rights reserved.
//|
//|    Redistribution and use in source and binary forms, with or without
//|    modification, are permitted provided that the following conditions are met:
//|
//|    1. Redistributions of source code must retain the above copyright notice, this
//|       list of conditions and the following disclaimer.
//|
//|    2. Redistributions in binary form must reproduce the above copyright notice,
//|       this list of conditions and the following disclaimer in the documentation
//|       and/or other materials provided with the distribution.
//|
//|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//|
#ifndef ALGEVO_ALGO_CEM_DISCRETE_HPP
#define ALGEVO_ALGO_CEM_DISCRETE_HPP

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
        class CrossEntropyMethodDiscrete {
        public:
            using population_t = Eigen::Matrix<unsigned int, -1, -1>;
            using x_t = Eigen::Matrix<unsigned int, -1, 1>;
            using p_t = Eigen::Matrix<Scalar, -1, -1>;
            using fit_t = Eigen::Matrix<Scalar, 1, -1>;
            using fit_eval_t = std::vector<Fit>;

            using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
            using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;

            struct Params {
                int seed = -1;

                unsigned int dim = 0;
                unsigned int pop_size = 0;
                unsigned int num_elites = 0;

                std::vector<unsigned int> num_values;
                p_t init_probs;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;

                x_t best;
                Scalar best_value;
            };

            CrossEntropyMethodDiscrete(const Params& params, const Fit& init_fit = {}) : _params(params), _update_coeff(static_cast<Scalar>(1.) / static_cast<Scalar>(_params.num_elites))
            {
                assert(_params.pop_size > 0 && "Population size needs to be bigger than zero!");
                assert(_params.dim > 0 && "Dimensions not set!");
                assert(_params.num_elites > 0 && _params.num_elites <= _params.pop_size && "Number of elites is wrongly set!");

                _allocate_data(init_fit);

                _fit_best = -std::numeric_limits<Scalar>::max();
            }

            IterationLog step()
            {
                // Generate population
                _generate_population();

                // Evaluate population
                _evaluate_population();

                // Update probabilities
                _update_distribution();

                // Update iteration log
                _log.iterations++;
                _log.func_evals += _params.pop_size;
                _log.best = _best;
                _log.best_value = _fit_best;

                return _log;
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

            const p_t& probabilities() const { return _probs; }

            const fit_t& population_fit() const { return _population_fit; }

            const x_t& best() const { return _best; }
            Scalar best_value() const { return _fit_best; }

        protected:
            // Parameters
            Params _params;

            // Iteration Log
            IterationLog _log;

            // Actual population (current values)
            population_t _population;
            fit_t _population_fit;

            // Categorical Distribution
            p_t _probs;

            // Data for updates
            const Scalar _update_coeff;
            population_t _elites;

            // Best ever
            x_t _best;
            Scalar _fit_best;

            // Evaluators
            fit_eval_t _fit_evals;

            void _allocate_data(const Fit& init_fit = {})
            {
                _population = population_t(_params.dim, _params.pop_size);
                _elites = population_t(_params.dim, _params.num_elites);

                _best = x_t::Constant(_params.dim, 0);
                _population_fit = fit_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());

                _probs = _params.init_probs;

                _fit_evals.resize(_params.pop_size, init_fit);
            }

            void _generate_population()
            {
                static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);

                // Generate random gaussian values from pure Normal distribution (mean=0, std=1)
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        Scalar p = rgen.rand();
                        Scalar s = static_cast<Scalar>(0.);
                        unsigned int k = 0;
                        for (; k < _params.num_values[j]; k++) {
                            s += _probs(j, k);
                            if (p < s)
                                break;
                        }
                        _population(j, i) = k;
                    }
                }
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

                // Update probabilities using the elites!
                for (unsigned int j = 0; j < _params.dim; j++) {
                    std::vector<unsigned int> counter(_params.num_values[j], 0);
                    for (unsigned int i = 0; i < _params.num_elites; i++) {
                        counter[_elites(j, i)]++;
                    }
                    for (unsigned int k = 0; k < _params.num_values[j]; k++) {
                        _probs(j, k) = static_cast<Scalar>(counter[k]) / static_cast<Scalar>(_params.num_elites);
                    }
                }
            }
        };
    } // namespace algo
} // namespace algevo

#endif
