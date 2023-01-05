#ifndef ALGEVO_ALGO_PSO_GRAD_HPP
#define ALGEVO_ALGO_PSO_GRAD_HPP

#include <Eigen/Core>

#include <array>
#include <limits>

#include <algevo/tools/parallel.hpp>
#include <algevo/tools/random.hpp>

#include <proxsuite/proxqp/dense/dense.hpp>

namespace algevo {
    namespace algo {
        template <typename Fit, typename Scalar = double>
        class ParticleSwarmOptimizationGrad {
        public:
            using population_t = Eigen::Matrix<Scalar, -1, -1>;
            using mat_t = population_t;
            using x_t = Eigen::Matrix<Scalar, -1, 1>;

            using fit_eval_t = std::vector<Fit>;

            using qp_t = proxsuite::proxqp::dense::QP<Scalar>;
            using qp_array_t = std::vector<std::unique_ptr<qp_t>>;
            using qp_mat_t = proxsuite::proxqp::dense::Mat<Scalar>;
            using qp_vec_t = proxsuite::proxqp::dense::Vec<Scalar>;

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
                Scalar qp_cr = static_cast<Scalar>(0.5);
                Scalar qp_alpha = static_cast<Scalar>(1.);
                Scalar qp_weight = static_cast<Scalar>(1.);
                Scalar epsilon_comp = static_cast<Scalar>(1e-4);

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
                Scalar best_cv;
            };

            ParticleSwarmOptimizationGrad(const Params& params) : _params(params), _rgen(0., 1., params.seed)
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

                // Initialize QPs
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    _qp_population[i] = std::make_unique<qp_t>(_params.dim, _params.neq_dim, _params.nin_dim);
                    _qp_init[i] = false;
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
                _log.best_cv = _cv_best;

                return _log;
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

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

            // One QP object per particle
            qp_array_t _qp_population;
            std::vector<bool> _qp_init;
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
                _qp_population.resize(_params.pop_size);
                _qp_init.resize(_params.pop_size, false);
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, _params.pop_size, [this](size_t i) {
                    auto res = _fit_evals[i].eval_all(_population.col(i));
                    Scalar f = std::get<0>(res);
                    Scalar cv = std::get<6>(res);
                    if (cv < _cv_best_local[i] || (_is_equal(cv, _cv_best_local[i]) && f > _fit_best_local[i])) {
                        _fit_best_local[i] = f;
                        _cv_best_local[i] = cv;
                        _best_local.col(i) = _population.col(i);
                    }

                    // Cache last evaluation for particle update
                    _eval_data[i].value = f;
                    _eval_data[i].constraint_violation = cv;

                    _eval_data[i].grad = std::get<1>(res);
                    _eval_data[i].eq_violation = std::get<2>(res);
                    _eval_data[i].ineq_violation = std::get<4>(res);

                    _eval_data[i].grad_eq = std::get<3>(res);
                    _eval_data[i].grad_ineq = std::get<5>(res);
                });

                // Update neighborhood and global bests
                for (unsigned int i = 0; i < _params.pop_size; i++) {
                    if (_cv_best_local[i] < _cv_best || (_is_equal(_cv_best_local[i], _cv_best) && _fit_best_local[i] > _fit_best)) {
                        _fit_best = _fit_best_local[i];
                        _cv_best = _cv_best_local[i];
                        _best = _best_local.col(i); // TO-DO: Maybe tag to avoid copies?
                    }

                    if (_cv_best_local[i] < _cv_best_neighbor[_neighborhood_ids[i]] || (_is_equal(_cv_best_local[i], _cv_best_neighbor[_neighborhood_ids[i]]) && _fit_best_local[i] > _fit_best_neighbor[_neighborhood_ids[i]])) {
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
                Scalar qp_cr = _params.qp_cr;
                Scalar one_minus_u = static_cast<Scalar>(1.) - u;
                Scalar one_minus_qp_alpha = static_cast<Scalar>(1.) - _params.qp_alpha;

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

                // Compute LP-ish direction to move in
                if (_params.qp_alpha > zero && rgen.rand() < qp_cr) {
                    // Get cached variables
                    qp_mat_t H = qp_mat_t::Identity(_params.dim, _params.dim);
                    qp_vec_t g = -_eval_data[i].grad;

                    qp_mat_t A = _eval_data[i].grad_eq;
                    qp_vec_t b = -_eval_data[i].eq_violation;

                    qp_mat_t C = _eval_data[i].grad_ineq;
                    qp_vec_t l = -_eval_data[i].ineq_violation;

                    // _qp_population[i]->settings.eps_abs = 1e-3; // choose accuracy needed
                    _qp_population[i]->settings.max_iter = 20;
                    _qp_population[i]->settings.max_iter_in = 10;
                    // _qp_population[i]->settings.verbose = true;
                    _qp_population[i]->settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
                    if (_qp_init[i])
                        _qp_population[i]->settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
                    if (_params.neq_dim > 0 && _params.nin_dim > 0) {
                        if (!_qp_init[i])
                            _qp_population[i]->init(H, g, A, b, C, l, proxsuite::nullopt);
                        else
                            _qp_population[i]->update(H, g, A, b, C, l, proxsuite::nullopt);
                    }
                    else if (_params.neq_dim > 0) {
                        if (!_qp_init[i])
                            _qp_population[i]->init(H, g, A, b, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
                        else
                            _qp_population[i]->update(H, g, A, b, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
                    }
                    else if (_params.nin_dim > 0) {
                        if (!_qp_init[i])
                            _qp_population[i]->init(H, g, proxsuite::nullopt, proxsuite::nullopt, C, l, proxsuite::nullopt);
                        else
                            _qp_population[i]->update(H, g, proxsuite::nullopt, proxsuite::nullopt, C, l, proxsuite::nullopt);
                    }
                    else {
                        if (!_qp_init[i])
                            _qp_population[i]->init(H, g, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
                        else
                            _qp_population[i]->update(H, g, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
                    }
                    _qp_population[i]->solve();

                    // Update velocities only when QP is successfull
                    if (_qp_population[i]->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED || _qp_population[i]->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED) {
                        _qp_init[i] = true;
                        // if (i == 0) {
                        //     std::cout << _population.col(i) << std::endl;
                        //     std::cout << _params.qp_weight * _params.qp_alpha * _qp_population[i]->results.x.transpose() << std::endl;
                        //     std::cout << std::endl;
                        // }
                        // TO-DO: Maybe add Armijo? https://solmaz.eng.uci.edu/Teaching/MAE206/Lecture4.pdf
                        if (one_minus_qp_alpha > zero)
                            _velocities.col(i) = _params.qp_alpha * _params.qp_weight * _qp_population[i]->results.x.transpose() + one_minus_qp_alpha * _velocities.col(i);
                        else
                            _velocities.col(i) = _params.qp_weight * _qp_population[i]->results.x.transpose();
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

            bool _is_equal(double a, double b)
            {
                return std::abs(a - b) <= _params.epsilon_comp;
            }
        };
    } // namespace algo
} // namespace algevo

#endif
