#ifndef ALGEVO_ALGO_PSO_HPP
#define ALGEVO_ALGO_PSO_HPP

#include <iostream>

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        template <typename Params, typename Fit, typename Scalar = double>
        class ParticleSwarmOptimization {
        public:
            using population_t = Eigen::Matrix<Scalar, Params::pop_size, Params::dim>;
            using fit_t = Eigen::Matrix<Scalar, Params::pop_size, 1>;
            using neighborhoods_t = Eigen::Matrix<Scalar, Params::num_neighborhoods, Params::dim>;
            using neighborhood_fit_t = Eigen::Matrix<Scalar, Params::num_neighborhoods, 1>;
            using x_t = Eigen::Matrix<Scalar, 1, Params::dim>;

            using fit_eval_t = std::array<Fit, Params::pop_size>;

            using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
            using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;

            ParticleSwarmOptimization()
            {
                // for (unsigned int i = 0; i < Params::pop_size; i++) {
                //     for (unsigned int j = 0; j < Params::dim; j++) {
                tools::parallel_loop(0, Params::pop_size * Params::dim, [this](size_t k) {
                    size_t i = k / Params::dim;
                    size_t j = k % Params::dim;
                    _population(i, j) = _rgen.rand();
                    _velocities(i, j) = _rgen_vel.rand();
                });

                _fit_best_local = fit_t::Constant(-std::numeric_limits<Scalar>::max());
                _fit_best_neighbor = neighborhood_fit_t::Constant(-std::numeric_limits<Scalar>::max());

                _fit_best = -std::numeric_limits<Scalar>::max();

                int n_id = -1;
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    if ((i % Params::num_neighbors) == 0)
                        n_id++;
                    _neighborhood_ids[i] = n_id;
                }

                _evaluate_population(); // initial evaluation of population
            }

            void step()
            {
                // Do updates
                tools::parallel_loop(0, Params::pop_size, [this](size_t i) {
                    _update_particle(i);
                });

                _evaluate_population();
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

            const population_t& velocities() const { return _velocities; }
            population_t& velocities() { return _velocities; }

            const x_t& best() const { return _best; }
            Scalar best_value() const { return _fit_best; }

        protected:
            // Actual population (current values)
            population_t _population;
            population_t _velocities;

            // Best ever per particle
            fit_t _fit_best_local;
            population_t _best_local;

            // Best ever per neighborhood
            neighborhood_fit_t _fit_best_neighbor;
            neighborhoods_t _best_neighbor;

            // Best ever
            x_t _best;
            Scalar _fit_best;

            // Neighborhoods
            std::array<int, Params::pop_size> _neighborhood_ids;

            // Evaluators
            fit_eval_t _fit_evals;

            // Random numbers
            rgen_scalar_t _rgen = rgen_scalar_t(Params::min_value, Params::max_value);
            rgen_scalar_t _rgen_vel = rgen_scalar_t(Params::min_vel, Params::max_vel);

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, Params::pop_size, [this](size_t i) {
                    Scalar f = _fit_evals[i].eval(_population.row(i));
                    if (f > _fit_best_local[i]) {
                        _fit_best_local[i] = f;
                        _best_local.row(i) = _population.row(i);
                    }
                });

                // Update neighborhood and global bests
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    if (_fit_best_local[i] > _fit_best) {
                        _fit_best = _fit_best_local[i];
                        _best = _population.row(i); // TO-DO: Maybe tag to avoid copies?
                    }

                    if (_fit_best_local[i] > _fit_best_neighbor[_neighborhood_ids[i]]) {
                        _fit_best_neighbor[_neighborhood_ids[i]] = _fit_best_local[i];
                        _best_neighbor.row(_neighborhood_ids[i]) = _population.row(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }
            }

            void _update_particle(unsigned int i)
            {
                static thread_local rgen_scalar_t rgen(0., 1.);

                Scalar chi = Params::chi;
                Scalar c1 = Params::c1;
                Scalar c2 = Params::c2;
                Scalar u = Params::u;
                Scalar one_minus_u = static_cast<Scalar>(1) - u;

                Scalar r1 = rgen.rand();
                Scalar r2 = rgen.rand();

                Scalar r1p = rgen.rand();
                Scalar r2p = rgen.rand();

                _velocities.row(i) = u * chi * (_velocities.row(i) + c1 * r1 * (_best_local.row(i) - _population.row(i)) + c2 * r2 * (_best - _population.row(i))).array() + one_minus_u * chi * (_velocities.row(i) + c1 * r1p * (_best_local.row(i) - _population.row(i)) + c2 * r2p * (_best_neighbor.row(_neighborhood_ids[i]) - _population.row(i))).array();

                _population.row(i) += _velocities.row(i);
            }
        };
    } // namespace algo
} // namespace algevo

#endif
