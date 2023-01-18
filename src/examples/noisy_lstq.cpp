#include <iostream>

#include <algevo/algo/pso_grad.hpp>

#include <Eigen/Dense>

namespace global {
    using rdist_scalar_gauss_t = std::normal_distribution<double>;
    using rgen_scalar_gauss_t = algevo::tools::RandomGenerator<rdist_scalar_gauss_t>;

    Eigen::Matrix<double, -1, -1> _A;
    Eigen::Matrix<double, -1, 1> _b;
    Eigen::Matrix<double, -1, 1> _grad_bias;
    Eigen::Matrix<double, -1, 1> _optimal;
    double _f_optimal;
    rgen_scalar_gauss_t _rgen = rgen_scalar_gauss_t(0., 1.);
} // namespace global

template <typename Scalar = double>
struct NoisyLstSq {
    static constexpr unsigned int dim = 1000;
    static constexpr unsigned int dim_features = dim;
    static constexpr unsigned int nsamples = 2000;
    static constexpr double max_value = 1.;
    static constexpr double min_value = -1.;
    static constexpr double max_features_value = 1.;
    static constexpr double min_features_value = 0.;
    static constexpr unsigned int neq_dim = 0;
    static constexpr unsigned int nin_dim = 0;

    using x_t = Eigen::Matrix<Scalar, 1, dim>;
    using c_t = Eigen::Matrix<Scalar, 1, neq_dim>;
    using g_t = Eigen::Matrix<Scalar, 1, nin_dim>;
    using C_t = Eigen::Matrix<Scalar, -1, -1>;
    using G_t = Eigen::Matrix<Scalar, -1, -1>;

    using Vector = Eigen::Matrix<Scalar, -1, 1>;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;

    Scalar eval(const x_t& x)
    {
        return std::get<0>(eval_all(x));
    }

    std::tuple<Scalar, x_t, c_t, C_t, g_t, G_t, Scalar> eval_all(const x_t& x, bool verbose = false)
    {
        x_t grad;
        grad.setZero();
        c_t c;
        c.setZero();
        C_t C(neq_dim, dim);
        C.setZero();
        g_t g;
        g.setZero();
        G_t G(nin_dim, dim);
        G.setZero();

        Vector residual = global::_A * x.transpose() - global::_b;
        Scalar cost = 0.5 * residual.squaredNorm() / static_cast<Scalar>(nsamples);

        Vector err = global::_A.transpose() * residual / static_cast<Scalar>(nsamples);
        Vector grad_noise(dim);
        for (unsigned int j = 0; j < dim; j++) {
            grad_noise[j] = global::_rgen.rand();
        }
        grad_noise = 1.5 * grad_noise / grad_noise.norm();
        grad = err + (global::_grad_bias + grad_noise) * err.norm();

        Scalar cv = 0.;
        for (unsigned int i = 0; i < neq_dim; i++) {
            cv += std::abs(c[i]);
        }
        for (unsigned int i = 0; i < nin_dim; i++) {
            cv += std::abs(std::min(0., g[i]));
        }

        if (verbose) {
            std::cout << grad.norm() << std::endl;
            std::cout << cost << " -> " << cv << ": " << global::_f_optimal << std::endl;
        }

        return {-cost, -grad, c, C, g, G, cv};
    }
};

// Typedefs
using NoisyLSq = NoisyLstSq<double>;
using Algo = algevo::algo::ParticleSwarmOptimizationQP<NoisyLSq>;
using Params = Algo::Params;

int main()
{
    // Instantiate problem values
    {
        global::_A = NoisyLSq::Matrix::Zero(NoisyLSq::nsamples, NoisyLSq::dim);
        global::_b = NoisyLSq::Vector::Zero(NoisyLSq::nsamples);
        global::_grad_bias = NoisyLSq::Vector::Zero(NoisyLSq::dim);

        for (unsigned int i = 0; i < NoisyLSq::nsamples; i++) {
            global::_b[i] = global::_rgen.rand();
            for (unsigned int j = 0; j < NoisyLSq::dim; j++) {
                global::_A(i, j) = global::_rgen.rand();
            }
        }

        for (unsigned int j = 0; j < NoisyLSq::dim; j++) {
            global::_grad_bias[j] = global::_rgen.rand();
        }

        global::_grad_bias = global::_grad_bias / global::_grad_bias.norm();

        global::_optimal = global::_A.colPivHouseholderQr().solve(global::_b);
        global::_f_optimal = 0.5 * (global::_A * global::_optimal - global::_b).squaredNorm() / static_cast<double>(NoisyLSq::nsamples);
    }

    // Set parameters
    Params params;
    params.dim = NoisyLSq::dim;
    params.pop_size = 40;
    params.num_neighbors = 4;
    params.max_value = Algo::x_t::Constant(params.dim, NoisyLSq::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, NoisyLSq::min_value);
    params.max_vel = Algo::x_t::Constant(params.dim, 1.);
    params.min_vel = Algo::x_t::Constant(params.dim, -1.);
    params.qp_alpha = 1.;
    params.qp_cr = 0.05;
    params.neq_dim = NoisyLSq::neq_dim;
    params.nin_dim = NoisyLSq::nin_dim;

    // Instantiate algorithm
    Algo pso(params);

    // Custom population initialization for faster convergence
    algevo::tools::rgen_gauss_t rgen(0., 0.1);
    for (unsigned int k = 0; k < params.pop_size; k++) {
        pso.population().col(k).setZero();
        // pso.velocities().col(k).setZero();
        for (unsigned int i = 0; i < params.dim; i++) {
            pso.population()(i, k) += rgen.rand();
        }
    }

    // Run a few iterations!
    for (unsigned int i = 0; i < (20000 / params.pop_size); i++) {
        auto log = pso.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << -log.best_value << " vs " << global::_f_optimal << std::endl;
    }

    return 0;
}
