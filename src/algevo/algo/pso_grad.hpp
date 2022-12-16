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
        template <typename Params, typename Fit, typename Scalar = double>
        class ParticleSwarmOptimizationGrad {
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

            using qp_t = proxsuite::proxqp::dense::QP<Scalar>;
            using qp_array_t = std::array<std::unique_ptr<qp_t>, Params::pop_size>;
            using qp_mat_t = proxsuite::proxqp::dense::Mat<Scalar>;
            using qp_vec_t = proxsuite::proxqp::dense::Vec<Scalar>;

            struct EvalInfo {
                Scalar value;
                Scalar constraint_violation;
                Eigen::Matrix<Scalar, -1, 1> grad;
                Eigen::Matrix<Scalar, -1, 1> eq_violation;
                Eigen::Matrix<Scalar, -1, -1> grad_eq;
                Eigen::Matrix<Scalar, -1, 1> ineq_violation;
                Eigen::Matrix<Scalar, -1, -1> grad_ineq;
            };

            using eval_info_t = std::array<EvalInfo, Params::pop_size>;

            using rdist_scalar_t = std::uniform_real_distribution<Scalar>;
            using rgen_scalar_t = tools::RandomGenerator<rdist_scalar_t>;
            using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
            using rgen_scalar_gauss_t = tools::RandomGenerator<rdist_scalar_gauss_t>;

            ParticleSwarmOptimizationGrad()
            {
                static_assert(Params::num_neighbors <= Params::pop_size, "Neighbor size needs to be smaller than population!");
                _allocate_data();

                // Initialize population
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    for (unsigned int j = 0; j < Params::dim; j++) {
                        _population(i, j) = _rgen.rand();
                        _velocities(i, j) = _rgen_vel.rand();
                    }
                }

                // Initial value
                _fit_best = -std::numeric_limits<Scalar>::max();
                _cv_best = std::numeric_limits<Scalar>::max();
                _nfe = 0;

                // Initialize neighbors
                int n_id = -1;
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    if ((i % Params::num_neighbors) == 0)
                        n_id++;
                    _neighborhood_ids[i] = std::min(n_id, static_cast<int>(Params::num_neighborhoods) - 1);
                }

                // Initialize QPs
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    _qp_population[i] = std::make_unique<qp_t>(Params::dim, Params::neq_dim, Params::nin_dim);
                    _qp_init[i] = false;
                }
            }

            void step()
            {
                // Evaluate population
                _evaluate_population();

                // Do updates
                tools::parallel_loop(0, Params::pop_size, [this](size_t i) {
                    _update_particle(i);
                });

                _nfe += Params::pop_size;
            }

            const population_t& population() const { return _population; }
            population_t& population() { return _population; }

            const population_t& velocities() const { return _velocities; }
            population_t& velocities() { return _velocities; }

            const x_t& best() const { return _best; }
            Scalar best_value() const { return _fit_best; }

            unsigned int nfe() const { return _nfe; }

        protected:
            // Actual population (current values)
            population_t _population;
            population_t _velocities;

            // One QP object per particle
            qp_array_t _qp_population;
            std::array<bool, Params::pop_size> _qp_init;
            eval_info_t _eval_info;

            // Best ever per particle
            fit_t _fit_best_local;
            fit_t _cv_best_local;
            population_t _best_local;

            // Best ever per neighborhood
            neighborhood_fit_t _fit_best_neighbor;
            neighborhood_fit_t _cv_best_neighbor;
            neighborhoods_t _best_neighbor;

            // Best ever
            x_t _best;
            Scalar _fit_best;
            Scalar _cv_best;

            // Neighborhoods
            std::array<int, Params::pop_size> _neighborhood_ids;

            // Evaluators
            fit_eval_t _fit_evals;

            // Number of Function Evaluations
            unsigned int _nfe;

            // Random numbers
            rgen_scalar_t _rgen = rgen_scalar_t(Params::min_value, Params::max_value, Params::seed);
            rgen_scalar_t _rgen_vel = rgen_scalar_t(Params::min_vel, Params::max_vel, Params::seed);

            void _allocate_data()
            {
                _population = population_t(Params::pop_size, Params::dim);
                _velocities = population_t(Params::pop_size, Params::dim);

                _fit_best_local = fit_t::Constant(Params::pop_size, -std::numeric_limits<Scalar>::max());
                _cv_best_local = fit_t::Constant(Params::pop_size, std::numeric_limits<Scalar>::max());

                _best_local = population_t(Params::pop_size, Params::dim);

                _fit_best_neighbor = neighborhood_fit_t::Constant(Params::num_neighborhoods, -std::numeric_limits<Scalar>::max());
                _cv_best_neighbor = neighborhood_fit_t::Constant(Params::num_neighborhoods, std::numeric_limits<Scalar>::max());
                _best_neighbor = neighborhoods_t(Params::num_neighborhoods, Params::dim);

                _best = x_t::Constant(Params::dim, -std::numeric_limits<Scalar>::max());

                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    _eval_info[i].value = 0.;
                    _eval_info[i].constraint_violation = 0.;

                    _eval_info[i].grad = Eigen::Matrix<Scalar, -1, 1>::Zero(Params::dim);
                    _eval_info[i].eq_violation = Eigen::Matrix<Scalar, -1, 1>::Zero(Params::neq_dim);
                    _eval_info[i].ineq_violation = Eigen::Matrix<Scalar, -1, 1>::Zero(Params::nin_dim);

                    _eval_info[i].grad_eq = Eigen::Matrix<Scalar, -1, -1>::Zero(Params::neq_dim, Params::dim);
                    _eval_info[i].grad_ineq = Eigen::Matrix<Scalar, -1, -1>::Zero(Params::nin_dim, Params::dim);
                }
            }

            void _evaluate_population()
            {
                // Evaluate individuals
                tools::parallel_loop(0, Params::pop_size, [this](size_t i) {
                    auto res = _fit_evals[i].eval_all(_population.row(i));
                    Scalar f = std::get<0>(res);
                    Scalar cv = std::get<6>(res);
                    if (cv < _cv_best_local[i] || (_is_equal(cv, _cv_best_local[i]) && f > _fit_best_local[i])) {
                        _fit_best_local[i] = f;
                        _cv_best_local[i] = cv;
                        _best_local.row(i) = _population.row(i);
                    }

                    // Cache last evaluation for particle update
                    _eval_info[i].value = f;
                    _eval_info[i].constraint_violation = cv;

                    _eval_info[i].grad = std::get<1>(res);
                    _eval_info[i].eq_violation = std::get<2>(res);
                    _eval_info[i].ineq_violation = std::get<4>(res);

                    _eval_info[i].grad_eq = std::get<3>(res);
                    _eval_info[i].grad_ineq = std::get<5>(res);
                });

                // Update neighborhood and global bests
                for (unsigned int i = 0; i < Params::pop_size; i++) {
                    if (_cv_best_local[i] < _cv_best || (_is_equal(_cv_best_local[i], _cv_best) && _fit_best_local[i] > _fit_best)) {
                        _fit_best = _fit_best_local[i];
                        _cv_best = _cv_best_local[i];
                        _best = _best_local.row(i); // TO-DO: Maybe tag to avoid copies?
                    }

                    if (_cv_best_local[i] < _cv_best_neighbor[_neighborhood_ids[i]] || (_is_equal(_cv_best_local[i], _cv_best_neighbor[_neighborhood_ids[i]]) && _fit_best_local[i] > _fit_best_neighbor[_neighborhood_ids[i]])) {
                        _fit_best_neighbor[_neighborhood_ids[i]] = _fit_best_local[i];
                        _cv_best_neighbor[_neighborhood_ids[i]] = _cv_best_local[i];
                        _best_neighbor.row(_neighborhood_ids[i]) = _best_local.row(i); // TO-DO: Maybe tag to avoid copies?
                    }
                }
            }

            void _update_particle(unsigned int i)
            {
                static thread_local rgen_scalar_t rgen(static_cast<Scalar>(0.), static_cast<Scalar>(1.), Params::seed);
                static thread_local rgen_scalar_gauss_t rgen_gauss(Params::mu_noise, Params::sigma_noise, Params::seed);

                static constexpr Scalar zero = static_cast<Scalar>(0.);
                static constexpr Scalar chi = Params::chi;
                static constexpr Scalar c1 = Params::c1;
                static constexpr Scalar c2 = Params::c2;
                static constexpr Scalar u = Params::u;
                static constexpr Scalar qp_cr = Params::qp_cr;
                static constexpr Scalar one_minus_u = static_cast<Scalar>(1.) - u;
                static constexpr Scalar one_minus_qp_alpha = static_cast<Scalar>(1.) - Params::qp_alpha;

                Scalar r1 = rgen.rand();
                Scalar r2 = rgen.rand();

                Scalar r1p = rgen.rand();
                Scalar r2p = rgen.rand();

                // In theory, those checks should be able to be defined at compile time and thus use the optimized version
                if (u > zero && one_minus_u > zero) // UPSO
                    _velocities.row(i) = one_minus_u * chi * (_velocities.row(i) + c1 * r1 * (_best_local.row(i) - _population.row(i)) + c2 * r2 * (_best - _population.row(i))).array() + u * chi * (_velocities.row(i) + c1 * r1p * (_best_local.row(i) - _population.row(i)) + c2 * r2p * (_best_neighbor.row(_neighborhood_ids[i]) - _population.row(i))).array();
                else if (u > zero) // GPSO
                    _velocities.row(i) = chi * (_velocities.row(i) + c1 * r1p * (_best_local.row(i) - _population.row(i)) + c2 * r2p * (_best_neighbor.row(_neighborhood_ids[i]) - _population.row(i)));
                else // if (one_minus_u > zero) // LPSO
                    _velocities.row(i) = chi * (_velocities.row(i) + c1 * r1 * (_best_local.row(i) - _population.row(i)) + c2 * r2 * (_best - _population.row(i)));

                // Add noise if wanted, helps to get away from local minima
                if (Params::noisy_velocity) {
                    for (unsigned int j = 0; j < Params::dim; j++) {
                        _velocities(i, j) += rgen_gauss.rand();
                    }
                }

                // Compute LP-ish direction to move in
                if (Params::qp_alpha > zero && rgen.rand() < qp_cr) {
                    // Get cached variables
                    qp_mat_t H = qp_mat_t::Identity(Params::dim, Params::dim);
                    qp_vec_t g = -_eval_info[i].grad;

                    qp_mat_t A = _eval_info[i].grad_eq;
                    qp_vec_t b = -_eval_info[i].eq_violation;

                    qp_mat_t C = _eval_info[i].grad_ineq;
                    qp_vec_t l = -_eval_info[i].ineq_violation;

                    // _qp_population[i]->settings.eps_abs = 1e-3; // choose accuracy needed
                    _qp_population[i]->settings.max_iter = 20;
                    _qp_population[i]->settings.max_iter_in = 10;
                    // _qp_population[i]->settings.verbose = true;
                    _qp_population[i]->settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
                    if (_qp_init[i])
                        _qp_population[i]->settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
                    if (Params::neq_dim > 0 && Params::nin_dim > 0) {
                        if (!_qp_init[i])
                            _qp_population[i]->init(H, g, A, b, C, l, proxsuite::nullopt);
                        else
                            _qp_population[i]->update(H, g, A, b, C, l, proxsuite::nullopt);
                    }
                    else if (Params::neq_dim > 0) {
                        if (!_qp_init[i])
                            _qp_population[i]->init(H, g, A, b, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
                        else
                            _qp_population[i]->update(H, g, A, b, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
                    }
                    else if (Params::nin_dim > 0) {
                        if (!_qp_init[i])
                            _qp_population[i]->init(H, g, proxsuite::nullopt, proxsuite::nullopt, C, l, proxsuite::nullopt);
                        else
                            _qp_population[i]->update(H, g, proxsuite::nullopt, proxsuite::nullopt, C, l, proxsuite::nullopt);
                    }
                    _qp_population[i]->solve();

                    // Update velocities only when QP is successfull
                    if (_qp_population[i]->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED || _qp_population[i]->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED) {
                        _qp_init[i] = true;
                        // if (i == 0) {
                        //     std::cout << _population.row(i) << std::endl;
                        //     std::cout << Params::qp_weight * Params::qp_alpha * _qp_population[i]->results.x.transpose() << std::endl;
                        //     std::cout << std::endl;
                        // }
                        // TO-DO: Maybe add Armijo? https://solmaz.eng.uci.edu/Teaching/MAE206/Lecture4.pdf
                        if (one_minus_qp_alpha > zero)
                            _velocities.row(i) = Params::qp_alpha * _qp_population[i]->results.x.transpose() + one_minus_qp_alpha * _velocities.row(i);
                        else
                            _velocities.row(i) = _qp_population[i]->results.x.transpose();
                    }
                }

                // Clamp velocities inside min/max
                for (unsigned int j = 0; j < Params::dim; j++) {
                    _velocities(i, j) = std::max(Params::min_vel, std::min(Params::max_vel, _velocities(i, j)));
                }

                // Update population
                _population.row(i) += _velocities.row(i);

                // Clamp inside min/max
                for (unsigned int j = 0; j < Params::dim; j++) {
                    _population(i, j) = std::max(Params::min_value, std::min(Params::max_value, _population(i, j)));
                }
            }

            static bool _is_equal(double a, double b, double eps = Params::epsilon_comp)
            {
                return std::abs(a - b) < eps;
            }
        };
    } // namespace algo
} // namespace algevo

#endif