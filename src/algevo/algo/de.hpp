//|
//|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
//|    Authors:  Konstantinos Chatzilygeroudis
//|    email:    costashatz@gmail.com
//|    website:  https://nosalro.github.io/
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
#ifndef ALGEVO_ALGO_DE_HPP
#define ALGEVO_ALGO_DE_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        template <typename Fit, typename Scalar = double>
        class DifferentialEvolution {
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

                x_t min_value;
                x_t max_value;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;

                x_t best;
                Scalar best_value;
            };

            DifferentialEvolution(const Params& params) : _params(params), _rgen(0., 1., params.seed)
            {
                assert(_params.pop_size >= 3 && "Population size needs to be bigger than 2!");
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
                // Do updates (evaluates new candidates as well)
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    _update_candidate(i);
                });

                // Update global best
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if (_population_fit(i) > _fit_best) {
                        _fit_best = _population_fit(i);
                        _best = _population.col(i); // TO-DO: Maybe tag to avoid copies?
                    }
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

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_t _rgen;

            void _allocate_data()
            {
                _population = population_t(_params.dim, _params.pop_size);
                _population_fit = x_t(_params.pop_size);
                _best = x_t(_params.dim);

                _fit_evals.resize(_params.pop_size);
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    _population_fit(i) = _fit_evals[i].eval(_population.col(i));
                });
            }

            void _update_candidate(unsigned int i)
            {
                static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);
                static thread_local tools::rgen_int_t rgen_dim(0, _params.dim - 1, _params.seed);
                static thread_local tools::rgen_int_t rgen_pop(0, _params.pop_size - 1, _params.seed);

                Scalar cr = _params.cr;
                Scalar f = _params.f;
                Scalar l = _params.lambda;

                // DE/rand-to best/1
                unsigned int i1 = rgen_pop.rand(), i2 = rgen_pop.rand();

                // Sample 2 distinct candidates
                while (i1 == i) {
                    i1 = rgen_pop.rand();
                }

                while (i2 == i || i2 == i1) {
                    i2 = rgen_pop.rand();
                }

                unsigned int R = rgen_dim.rand();

                x_t y = _population.col(i); // copy original candidate
                for (unsigned int j = 0; j < _params.dim; j++) {
                    if (j == R || rgen.rand() < cr) {
                        Scalar v = 0.;
                        if (_log.iterations > 0)
                            v = l * (_best(j) - _population(j, i));
                        y(j) = std::min(_params.max_value[j], std::max(_params.min_value[j], _population(j, i) + v + f * (_population(j, i1) - _population(j, i2))));
                    }
                }

                Scalar perf = _fit_evals[i].eval(y);
                if (_log.iterations == 0 || perf >= _population_fit(i)) {
                    _population_fit(i) = perf;
                    _population.col(i) = y;
                }
            }
        };
    } // namespace algo
} // namespace algevo

#endif
