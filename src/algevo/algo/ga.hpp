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
#ifndef ALGEVO_ALGO_GA_HPP
#define ALGEVO_ALGO_GA_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        template <typename Fit, typename Scalar = double>
        class GeneticAlgorithm {
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

                x_t min_value;
                x_t max_value;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;

                x_t best;
                Scalar best_value;
            };

            GeneticAlgorithm(const Params& params) : _params(params), _num_elites(std::min(params.num_elites, params.pop_size / 2)), _rgen(0., 1., params.seed)
            {
                assert(_params.pop_size >= 2 && "Population size needs to be bigger than 1!");
                assert(_params.dim > 0 && "Dimensions not set!");
                assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");
                _allocate_data();

                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        Scalar range = (_params.max_value[j] - _params.min_value[j]);
                        _population(j, i) = _rgen.rand() * range + _params.min_value[j];
                    }
                }

                _fit_best = -std::numeric_limits<Scalar>::max();
            }

            IterationLog step()
            {
                // Evaluate population
                _evaluate_population();

                // Update global best
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if (_population_fit(i) > _fit_best) {
                        _fit_best = _population_fit(i);
                        _best = _population.col(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }

                // Sort the population according to fitness
                _sort_population();

                // Perform mutation and crossover
                _genetic_operators();

                // Update iteration log
                _log.iterations++;
                _log.func_evals += _params.pop_size;
                _log.best = _best;
                _log.best_value = _fit_best;

                return _log;
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

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

            // Latest fitness evaluations
            x_t _population_fit;

            // Best ever
            x_t _best;
            Scalar _fit_best;

            // Indices for best/worst
            idx_t _best_idxs;

            unsigned int _num_elites;

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_t _rgen;
            rgen_scalar_gauss_t _rgen_gauss = rgen_scalar_gauss_t(static_cast<Scalar>(0.), static_cast<Scalar>(1.));

            void _allocate_data()
            {
                _population = population_t(_params.dim, _params.pop_size);
                _population_fit = x_t(_params.pop_size);
                _best = x_t(_params.dim);

                _fit_evals.resize(_params.pop_size);

                _best_idxs.resize(_params.pop_size);
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    _population_fit(i) = _fit_evals[i].eval(_population.col(i));
                });
            }

            void _sort_population()
            {
                // initialize original index locations
                std::iota(_best_idxs.begin(), _best_idxs.end(), 0);

                // sort indexes based on comparing fitness
                std::sort(_best_idxs.begin(), _best_idxs.end(), [this](size_t i1, size_t i2) { return _population_fit(i1) > _population_fit(i2); });
            }

            void _genetic_operators()
            {
                // Perform mutation and crossover on best and replace worst
                tools::parallel_loop(0, _num_elites, [this](size_t i) {
                    _mutation_and_crossover(i);
                });
            }

            void _mutation_and_crossover(unsigned int elite_idx)
            {
                static thread_local tools::rgen_int_t rgen_elites(0, _num_elites - 1, _params.seed);

                // mutation and crossover
                unsigned int p1 = _best_idxs[elite_idx];
                unsigned int p2 = _best_idxs[rgen_elites.rand()];

                // Make sure we always select a different elite. TO-DO: Is this really needed?
                while (p1 == p2) {
                    p2 = _best_idxs[rgen_elites.rand()];
                }

                // crossover
                _population.col(_best_idxs[_params.pop_size - elite_idx - 1]) = _population.col(p1) + _params.sigma_2 * _rgen_gauss.rand() * (_population.col(p2) - _population.col(p1));

                // Gaussian mutation
                for (unsigned int j = 0; j < _params.dim; j++) {
                    _population(j, _best_idxs[_params.pop_size - elite_idx - 1]) += _rgen_gauss.rand() * _params.sigma_1;
                    // clip in [min,max]
                    _population(j, _best_idxs[_params.pop_size - elite_idx - 1]) = std::max(_params.min_value[j], std::min(_params.max_value[j], _population(j, _best_idxs[_params.pop_size - elite_idx - 1])));
                }
            }
        };
    } // namespace algo
} // namespace algevo

#endif
