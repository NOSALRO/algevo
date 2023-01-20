#ifndef ALGEVO_ALGO_PSO_PENALTY_HPP
#define ALGEVO_ALGO_PSO_PENALTY_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

namespace algevo {
    namespace algo {
        template <typename Fit, typename Scalar = double>
        class ParticleSwarmOptimizationPenalty {
        public:
            using population_t = Eigen::Matrix<Scalar, -1, -1>;
            using mat_t = population_t;
            using x_t = Eigen::Matrix<Scalar, -1, 1>;

            using fit_eval_t = std::vector<Fit>;

            struct EvalData {
                Scalar value;
                Scalar constraint_violation;
                x_t grad;
                x_t eq_violation;
                mat_t grad_eq;
                x_t ineq_violation;
                mat_t grad_ineq;
            };

            using eval_data_t = std::vector<EvalData>;

            using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
            using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
            using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
            using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;

            struct Params {
                int seed = -1;

                unsigned int dim = 0;
                unsigned int pop_size = 0;
                unsigned int num_neighbors = 0;
                unsigned int neq_dim = 0;
                unsigned int nin_dim = 0;

                Scalar chi = static_cast<Scalar>(0.729);
                Scalar c1 = static_cast<Scalar>(2.05);
                Scalar c2 = static_cast<Scalar>(2.05);
                Scalar u = static_cast<Scalar>(0.5);

                bool noisy_velocity = true;
                Scalar mu_noise = static_cast<Scalar>(0.);
                Scalar sigma_noise = static_cast<Scalar>(0.0001);

                bool reevaluate_best = false; // re-evaluate best particles, turn on in noisy settings
                double forgetting_factor = 0.2;

                x_t min_value;
                x_t max_value;

                x_t min_vel;
                x_t max_vel;

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

            ParticleSwarmOptimizationPenalty(const Params& params) : _params(params), _rgen(0., 1., params.seed)
            {
                assert(_params.pop_size > 0 && "Population size needs to be bigger than zero!");
                assert(_params.dim > 0 && "Dimensions not set!");
                assert(_params.num_neighbors <= _params.pop_size && "Neighbor size needs to be smaller than population!");
                assert(_params.num_neighbors > 0 && "Neighbor size not set!");
                assert(_params.min_value.size() == _params.dim && _params.max_value.size() == _params.dim && "Min/max values dimensions should be the same as the problem dimensions!");
                assert(_params.min_vel.size() == _params.dim && _params.max_vel.size() == _params.dim && "Min/max velocity dimensions should be the same as the problem dimensions!");

                _num_neighborhoods = std::floor(_params.pop_size / static_cast<Scalar>(_params.num_neighbors));

                _allocate_data();

                // Initialize population
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    for (unsigned int j = 0; j < _params.dim; j++) {
                        Scalar range = (_params.max_value[j] - _params.min_value[j]);
                        Scalar range_vel = (_params.max_vel[j] - _params.min_vel[j]);
                        _population(j, i) = _rgen.rand() * range + _params.min_value[j];
                        _velocities(j, i) = _rgen.rand() * range_vel + _params.min_vel[j];
                    }
                }

                // Initial values
                _fit_best = -std::numeric_limits<Scalar>::max();
                _cv_best = std::numeric_limits<Scalar>::max();

                // Initialize neighbors
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
                if (_params.reevaluate_best && _log.iterations > 0) {
                    _log.func_evals += _params.pop_size; // we re-evaluated the best_local
                    _log.func_evals += 1; // we re-evaluated the best ever
                    _log.func_evals += _num_neighborhoods; // we re-evaluated the best ever per neighbor
                }
                _log.best = _best;
                _log.best_value = _fit_best;
                _log.best_cv = _cv_best;

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

            // Evaluation data
            eval_data_t _eval_data;

            // Best ever per particle
            x_t _fit_best_local;
            x_t _cv_best_local;
            population_t _best_local;

            // Best ever per neighborhood
            x_t _fit_best_neighbor;
            x_t _cv_best_neighbor;
            population_t _best_neighbor;

            // Best ever
            x_t _best;
            Scalar _fit_best;
            Scalar _cv_best;

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
                _cv_best_local = x_t::Constant(_params.pop_size, std::numeric_limits<Scalar>::max());

                _best_local = population_t(_params.dim, _params.pop_size);

                _fit_best_neighbor = x_t::Constant(_num_neighborhoods, -std::numeric_limits<Scalar>::max());
                _cv_best_neighbor = x_t::Constant(_num_neighborhoods, std::numeric_limits<Scalar>::max());
                _best_neighbor = population_t(_params.dim, _num_neighborhoods);

                _best = x_t::Constant(_params.dim, -std::numeric_limits<Scalar>::max());

                _eval_data.resize(_params.pop_size);
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    _eval_data[i].value = 0.;
                    _eval_data[i].constraint_violation = 0.;

                    _eval_data[i].grad = x_t::Zero(_params.dim);
                    _eval_data[i].eq_violation = x_t::Zero(_params.neq_dim);
                    _eval_data[i].ineq_violation = x_t::Zero(_params.nin_dim);

                    _eval_data[i].grad_eq = mat_t::Zero(_params.neq_dim, _params.dim);
                    _eval_data[i].grad_ineq = mat_t::Zero(_params.nin_dim, _params.dim);
                }

                _fit_evals.resize(_params.pop_size);
                _eval_data.resize(_params.pop_size);
                _neighborhood_ids.resize(_params.pop_size, -1);
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    auto res = _fit_evals[i].eval_all(_population.col(i));

                    // Cache last evaluation for particle update
                    _eval_data[i].value = std::get<0>(res);
                    _eval_data[i].constraint_violation = std::get<6>(res);

                    _eval_data[i].grad = std::get<1>(res);
                    _eval_data[i].eq_violation = std::get<2>(res);
                    _eval_data[i].ineq_violation = std::get<4>(res);

                    _eval_data[i].grad_eq = std::get<3>(res);
                    _eval_data[i].grad_ineq = std::get<5>(res);

                    Scalar f = _eval_data[i].value;
                    Scalar cv = _compute_penalty(_eval_data[i]);

                    // Re-evaluate best ever if needed (noisy settings)
                    if (_params.reevaluate_best && _log.iterations > 0) {
                        auto res_best = _fit_evals[i].eval_all(_best_local.col(i));
                        EvalData data;
                        data.eq_violation = std::get<2>(res_best);
                        data.ineq_violation = std::get<4>(res_best);

                        Scalar pen = _compute_penalty(data);

                        _fit_best_local[i] = _fit_best_local[i] + _params.forgetting_factor * (std::get<0>(res_best) - _fit_best_local[i]);
                        _cv_best_local[i] = _cv_best_local[i] + _params.forgetting_factor * (pen - _cv_best_local[i]);
                    }

                    if (_compare(f, cv, _fit_best_local[i], _cv_best_local[i], (_log.iterations + 1) * std::sqrt(_log.iterations + 1))) {
                        _fit_best_local[i] = f;
                        _cv_best_local[i] = cv;
                        _best_local.col(i) = _population.col(i);
                    }
                });

                // Update neighborhood and global bests
                // Re-evaluate best ever if needed (noisy settings)
                if (_params.reevaluate_best && _log.iterations > 0) {
                    // best ever
                    {
                        auto res_best = _fit_evals[0].eval_all(_best);
                        EvalData data;
                        data.eq_violation = std::get<2>(res_best);
                        data.ineq_violation = std::get<4>(res_best);

                        Scalar pen = _compute_penalty(data);

                        _fit_best = _fit_best + _params.forgetting_factor * (std::get<0>(res_best) - _fit_best);
                        _cv_best = _cv_best + _params.forgetting_factor * (pen - _cv_best);
                    }

                    // best ever per neighborhood
                    for (unsigned int i = 0; i < _num_neighborhoods; i++) {
                        auto res_best = _fit_evals[0].eval_all(_best_neighbor.col(i));
                        EvalData data;
                        data.eq_violation = std::get<2>(res_best);
                        data.ineq_violation = std::get<4>(res_best);

                        Scalar pen = _compute_penalty(data);

                        _fit_best_neighbor[i] = _fit_best_neighbor[i] + _params.forgetting_factor * (std::get<0>(res_best) - _fit_best_neighbor[i]);
                        _cv_best_neighbor[i] = _cv_best_neighbor[i] + _params.forgetting_factor * (pen - _cv_best_neighbor[i]);
                    }
                }

                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if (_compare(_fit_best_local[i], _cv_best_local[i], _fit_best, _cv_best, (_log.iterations + 1) * std::sqrt(_log.iterations + 1))) {
                        _fit_best = _fit_best_local[i];
                        _cv_best = _cv_best_local[i];
                        _best = _best_local.col(i); // TO-DO: Maybe tag to avoid copies?
                    }

                    if (_compare(_fit_best_local[i], _cv_best_local[i], _fit_best_neighbor[_neighborhood_ids[i]], _cv_best_neighbor[_neighborhood_ids[i]], (_log.iterations + 1) * std::sqrt(_log.iterations + 1))) {
                        _fit_best_neighbor[_neighborhood_ids[i]] = _fit_best_local[i];
                        _cv_best_neighbor[_neighborhood_ids[i]] = _cv_best_local[i];
                        _best_neighbor.col(_neighborhood_ids[i]) = _best_local.col(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }
            }

            void _update_particle(unsigned int i)
            {
                static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), _params.seed);
                static thread_local rgen_scalar_gauss_t rgen_gauss(_params.mu_noise, _params.sigma_noise, _params.seed);

                Scalar zero = static_cast<Scalar>(0.);
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

            Scalar _compute_penalty(const EvalData& data) const
            {
                Scalar pen = 0.;
                for (unsigned int i = 0; i < _params.neq_dim; i++) {
                    auto q = std::abs(data.eq_violation[i]);
                    pen += _theta(q) * q * std::pow(q, _gamma(q));
                }
                for (unsigned int i = 0; i < _params.nin_dim; i++) {
                    auto q = std::abs(std::min(0., data.ineq_violation[i]));
                    pen += _theta(q) * q * std::pow(q, _gamma(q));
                }

                return pen;
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

            bool _compare(Scalar f1, Scalar pen1, Scalar f2, Scalar pen2, Scalar h) const
            {
                Scalar v1 = -f1 + h * pen1;
                Scalar v2 = -f2 + h * pen2;

                return v1 < v2;
            }
        };
    } // namespace algo
} // namespace algevo

#endif
