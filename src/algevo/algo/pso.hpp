//|
//|    Copyright (c) 2022-2024 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2023-2024 Laboratory of Automation and Robotics, University of Patras, Greece
//|    Copyright (c) 2022-2024 Konstantinos Chatzilygeroudis
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
#ifndef ALGEVO_ALGO_PSO_HPP
#define ALGEVO_ALGO_PSO_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        template <typename Fit, typename Scalar = double>
        class ParticleSwarmOptimization {
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
                unsigned int num_neighbors = 0;

                Scalar chi = static_cast<Scalar>(0.729);
                Scalar c1 = static_cast<Scalar>(2.05);
                Scalar c2 = static_cast<Scalar>(2.05);
                Scalar u = static_cast<Scalar>(0.5);

                bool noisy_velocity = true;
                Scalar mu_noise = static_cast<Scalar>(0.);
                Scalar sigma_noise = static_cast<Scalar>(0.0001);

                x_t min_value;
                x_t max_value;

                x_t min_vel;
                x_t max_vel;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;

                x_t best;
                Scalar best_value;
            };

            ParticleSwarmOptimization(const Params& params) : _params(params), _rgen(0., 1., params.seed)
            {
                assert(_params.pop_size > 0 && "Population size needs to be bigger than zero!");
                assert(_params.dim > 0 && "Dimensions not set!");
                assert(_params.num_neighbors <= _params.pop_size && "Neighbor size needs to be smaller than population!");
                assert(_params.num_neighbors > 0 && "Neighbor size not set!");
                assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");
                assert(_params.min_vel.size() == _params.dim && _params.max_vel.size() == _params.dim && "Min/max velocity dimensions should be the same as the problem dimensions!");

                _num_neighborhoods = std::floor(_params.pop_size / static_cast<Scalar>(_params.num_neighbors));

                _allocate_data();

                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        Scalar range = (_params.max_value[j] - _params.min_value[j]);
                        Scalar range_vel = (_params.max_vel[j] - _params.min_vel[j]);
                        _population(j, i) = _rgen.rand() * range + _params.min_value[j];
                        _velocities(j, i) = _rgen.rand() * range_vel + _params.min_vel[j];
                    }
                }

                _fit_best = -std::numeric_limits<Scalar>::max();

                int n_id = -1;
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if ((i % _params.num_neighbors) == 0)
                        n_id++;
                    _neighborhood_ids[i] = std::min(n_id, static_cast<int>(_num_neighborhoods) - 1);
                }
            }

            IterationLog step()
            {
                // Evaluate population
                _evaluate_population();

                // Do updates
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    _update_particle(i);
                });

                // Update iteration log
                _log.iterations++;
                _log.func_evals += _params.pop_size;
                _log.best = _best;
                _log.best_value = _fit_best;

                return _log;
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

            x_t average() const { return _population.rowwise().mean(); }

            const population_t& velocities() const { return _velocities; }
            population_t& velocities() { return _velocities; }

            const x_t& best() const { return _best; }
            Scalar best_value() const { return _fit_best; }

        protected:
            // Parameters
            Params _params;

            // Iteration Log
            IterationLog _log;

            // Actual population (current values)
            population_t _population;
            population_t _velocities;

            // Best ever per particle
            x_t _fit_best_local;
            population_t _best_local;

            // Best ever per neighborhood
            x_t _fit_best_neighbor;
            population_t _best_neighbor;

            // Best ever
            x_t _best;
            Scalar _fit_best;

            // Neighborhoods
            std::vector<int> _neighborhood_ids;
            unsigned int _num_neighborhoods;

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_t _rgen;

            void _allocate_data()
            {
                _population = population_t(_params.dim, _params.pop_size);
                _velocities = population_t(_params.dim, _params.pop_size);

                _fit_best_local = x_t::Constant(_params.pop_size, -std::numeric_limits<Scalar>::max());

                _best_local = population_t(_params.dim, _params.pop_size);

                _fit_best_neighbor = x_t::Constant(_num_neighborhoods, -std::numeric_limits<Scalar>::max());
                _best_neighbor = population_t(_params.dim, _num_neighborhoods);

                _best = x_t::Constant(_params.dim, -std::numeric_limits<Scalar>::max());

                _fit_evals.resize(_params.pop_size);
                _neighborhood_ids.resize(_params.pop_size, -1);
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    Scalar f = _fit_evals[i].eval(_population.col(i));
                    if (f > _fit_best_local[i]) {
                        _fit_best_local[i] = f;
                        _best_local.col(i) = _population.col(i);
                    }
                });

                // Update neighborhood and global bests
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if (_fit_best_local[i] > _fit_best) {
                        _fit_best = _fit_best_local[i];
                        _best = _best_local.col(i); // TO-DO: Maybe tag to avoid copies?
                    }

                    if (_fit_best_local[i] > _fit_best_neighbor[_neighborhood_ids[i]]) {
                        _fit_best_neighbor[_neighborhood_ids[i]] = _fit_best_local[i];
                        _best_neighbor.col(_neighborhood_ids[i]) = _best_local.col(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }
            }

            void _update_particle(unsigned int i)
            {
                static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);
                static thread_local rgen_scalar_gauss_t rgen_gauss(_params.mu_noise, _params.sigma_noise, _params.seed);
                static Scalar zero = static_cast<Scalar>(0.);

                Scalar chi = _params.chi;
                Scalar c1 = _params.c1;
                Scalar c2 = _params.c2;
                Scalar u = _params.u;
                Scalar one_minus_u = static_cast<Scalar>(1.) - u;

                Scalar r1 = rgen.rand();
                Scalar r2 = rgen.rand();

                Scalar r1p = rgen.rand();
                Scalar r2p = rgen.rand();

                if (u > zero && one_minus_u > zero) // UPSO
                    _velocities.col(i) = one_minus_u * chi * (_velocities.col(i) + c1 * r1 * (_best_local.col(i) - _population.col(i)) + c2 * r2 * (_best - _population.col(i))).array() + u * chi * (_velocities.col(i) + c1 * r1p * (_best_local.col(i) - _population.col(i)) + c2 * r2p * (_best_neighbor.col(_neighborhood_ids[i]) - _population.col(i))).array();
                else if (u > zero) // GPSO
                    _velocities.col(i) = chi * (_velocities.col(i) + c1 * r1p * (_best_local.col(i) - _population.col(i)) + c2 * r2p * (_best_neighbor.col(_neighborhood_ids[i]) - _population.col(i)));
                else // if (one_minus_u > zero) // LPSO
                    _velocities.col(i) = chi * (_velocities.col(i) + c1 * r1 * (_best_local.col(i) - _population.col(i)) + c2 * r2 * (_best - _population.col(i)));

                // Add noise if wanted, helps to get away from local minima
                if (_params.noisy_velocity) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        _velocities(j, i) += rgen_gauss.rand();
                    }
                }

                // Clamp velocities inside min/max
                for (unsigned int j = 0; j < _params.dim; j++) {
                    _velocities(j, i) = std::max(_params.min_vel[j], std::min(_params.max_vel[j], _velocities(j, i)));
                }

                // Update population
                _population.col(i) += _velocities.col(i);

                // Clamp inside min/max
                for (unsigned int j = 0; j < _params.dim; j++) {
                    _population(j, i) = std::max(_params.min_value[j], std::min(_params.max_value[j], _population(j, i)));
                }
            }
        };
    } // namespace algo
} // namespace algevo

#endif
