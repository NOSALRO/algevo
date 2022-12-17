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
        // for (unsigned int i = 0; i < neq_dim; i++) {
        //     cv += std::abs(c[i]);
        // }
        // for (unsigned int i = 0; i < nin_dim; i++) {
        //     cv += std::abs(std::min(0., g[i]));
        // }

        if (verbose) {
            std::cout << grad.norm() << std::endl;
            std::cout << cost << " -> " << cv << ": " << global::_f_optimal << std::endl;
        }

        return {-cost, -grad, c, C, g, G, cv};
    }
};

using NoisyLSq = NoisyLstSq<double>;

struct ParamsPSO {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = NoisyLSq::dim;
    static constexpr unsigned int pop_size = 40;
    static constexpr unsigned int num_neighbors = 4;
    static constexpr unsigned int num_neighborhoods = std::floor(pop_size / static_cast<double>(num_neighbors));
    static constexpr double max_value = NoisyLSq::max_value;
    static constexpr double min_value = NoisyLSq::min_value;
    static constexpr double max_vel = 1.;
    static constexpr double min_vel = -1.;

    // Constraints
    static constexpr unsigned int neq_dim = NoisyLSq::neq_dim;
    static constexpr unsigned int nin_dim = NoisyLSq::nin_dim;

    static constexpr double chi = 0.729;
    static constexpr double c1 = 2.05;
    static constexpr double c2 = 2.05;
    static constexpr double u = 0.5;

    static constexpr bool noisy_velocity = true;
    static constexpr double mu_noise = 0.;
    static constexpr double sigma_noise = 0.0001;

    static constexpr double qp_alpha = 1.;
    static constexpr double qp_cr = 0.05;
    static constexpr double epsilon_comp = 1e-4;
};

int main()
{
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

    NoisyLSq s;
    algevo::algo::ParticleSwarmOptimizationGrad<ParamsPSO, NoisyLSq> pso;

    algevo::tools::rgen_gauss_t rgen(0., 0.1);
    for (unsigned int k = 0; k < ParamsPSO::pop_size; k++) {
        pso.population().row(k).setZero();
        // pso.velocities().row(k).setZero();
        for (unsigned int i = 0; i < ParamsPSO::dim; i++) {
            pso.population()(k, i) += rgen.rand();
        }
    }

    for (unsigned int i = 0; i < (10000 / ParamsPSO::pop_size); i++) {
        pso.step();
        // std::cout << i << ": " << pso.best_value() << std::endl;
        std::cout << pso.nfe() << " ";
        s.eval_all(pso.best(), true);
    }
    std::cout << pso.nfe() << " ";
    s.eval_all(pso.best(), true);
    // std::cout << "Best: " << pso.best() << std::endl;
    return 0;
}
