#include <iostream>

#include <algevo/algo/pso_grad.hpp>

// template <typename Scalar = double>
// struct SimpleExample {
//     static constexpr unsigned int dim = 2;
//     static constexpr unsigned int dim_features = dim;
//     static constexpr double max_value = 10.;
//     static constexpr double min_value = -10.;
//     static constexpr double max_features_value = 1.;
//     static constexpr double min_features_value = 0.;
//     // static constexpr unsigned int neq_dim = 1;
//     // static constexpr unsigned int nin_dim = 0;
//     static constexpr unsigned int neq_dim = 1;
//     static constexpr unsigned int nin_dim = 1;

//     using x_t = Eigen::Matrix<Scalar, 1, dim>;
//     using c_t = Eigen::Matrix<Scalar, 1, neq_dim>;
//     using g_t = Eigen::Matrix<Scalar, 1, nin_dim>;
//     using C_t = Eigen::Matrix<Scalar, neq_dim, dim>;
//     using G_t = Eigen::Matrix<Scalar, nin_dim, dim>;

//     using rdist_scalar_gauss_t = std::normal_distribution<Scalar>;
//     using rgen_scalar_gauss_t = algevo::tools::RandomGenerator<rdist_scalar_gauss_t>;

//     // Scalar eval(const x_t& x)
//     // {
//     //     return 49 - x[0] * x[0] - x[1] * x[1];
//     // }

//     // std::tuple<Scalar, x_t, c_t, C_t, g_t, G_t, Scalar> eval_all(const x_t& x, bool verbose = false)
//     // {
//     //     Scalar v = eval(x);
//     //     x_t grad;
//     //     grad << -2. * x[0], -2. * x[1];
//     //     c_t c;
//     //     c << x[0] + 3 * x[1] - 10.;
//     //     C_t C;
//     //     C << 1., 3.;
//     //     g_t g;
//     //     // g << x[0] - min_value, max_value - x[0], x[1] - min_value, max_value - x[1];
//     //     G_t G;
//     //     // G << 1., 0., -1., 0., 0., 1., 0., -1.;

//     //     Scalar cv = std::abs(c[0]);
//     //     for (unsigned int i = 0; i < nin_dim; i++) {
//     //         cv += std::abs(std::min(0., g[i]));
//     //     }

//     //     if (verbose) {
//     //         std::cout << v << " -> " << cv << std::endl;
//     //     }

//     //     return {v, grad, c, C, g, G, cv};
//     // }

//     Scalar eval(const x_t& x)
//     {
//         return -((x[0] - 2.) * (x[0] - 2.) + (x[1] - 1.) * (x[1] - 1.));
//     }

//     std::tuple<Scalar, x_t, c_t, C_t, g_t, G_t, Scalar> eval_all(const x_t& x, bool verbose = false)
//     {
//         static thread_local rgen_scalar_gauss_t rgen_gauss(0., 0.01, -1);
//         Scalar v = eval(x);
//         x_t grad;
//         grad << -2. * (x[0] - 2.), -2. * (x[1] - 1.);
//         c_t c;
//         c << x[0] - 2. * x[1] + 1.;
//         C_t C;
//         C << 1., -2.;
//         g_t g;
//         g << -(x[0] * x[0] / 4. + x[1] * x[1] - 1);
//         G_t G;
//         G << -x[0] / 2., -2. * x[1];

//         // if (!verbose) {
//         //     // add some noise
//         //     for (unsigned int i = 0; i < dim; i++) {
//         //         grad(i) += rgen_gauss.rand();

//         //         for (unsigned int j = 0; j < neq_dim; j++) {
//         //             C(j, i) += rgen_gauss.rand();
//         //         }

//         //         for (unsigned int j = 0; j < nin_dim; j++) {
//         //             G(j, i) += rgen_gauss.rand();
//         //         }
//         //     }

//         //     for (unsigned int j = 0; j < neq_dim; j++) {
//         //         c(j) += rgen_gauss.rand();
//         //     }

//         //     for (unsigned int j = 0; j < nin_dim; j++) {
//         //         g(j) += rgen_gauss.rand();
//         //     }
//         // }

//         Scalar cv = 0.;
//         for (unsigned int i = 0; i < neq_dim; i++) {
//             cv += std::abs(c[i]);
//         }
//         for (unsigned int i = 0; i < nin_dim; i++) {
//             cv += std::abs(std::min(0., g[i]));
//         }

//         if (verbose) {
//             std::cout << v << " -> " << cv << std::endl;
//         }

//         return {v, grad, c, C, g, G, cv};
//     }
// };

// using SimExp = SimpleExample<double>;

template <typename Scalar = double>
struct DoubleIntegrator {
    static constexpr unsigned int D = 2;
    static constexpr unsigned int Ad = 1;
    static constexpr unsigned int T = 51;
    double dt = 0.1;

    static constexpr unsigned int dim = T * Ad + (T - 1) * D;
    static constexpr unsigned int dim_features = dim;
    static constexpr double max_value = 100.;
    static constexpr double min_value = -100.;
    static constexpr double max_features_value = 1.;
    static constexpr double min_features_value = 0.;
    // static constexpr unsigned int neq_dim = 1;
    // static constexpr unsigned int nin_dim = 0;
    static constexpr unsigned int neq_dim = T * D;
    static constexpr unsigned int nin_dim = 0; // 2 * dim;

    using x_t = Eigen::Matrix<Scalar, 1, dim>;
    using c_t = Eigen::Matrix<Scalar, 1, neq_dim>;
    using g_t = Eigen::Matrix<Scalar, 1, nin_dim>;
    using C_t = Eigen::Matrix<Scalar, -1, -1>;
    using G_t = Eigen::Matrix<Scalar, -1, -1>;

    using Vector = Eigen::Matrix<Scalar, -1, 1>;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;

    std::tuple<Scalar, x_t, c_t, C_t, g_t, G_t, Scalar> eval_all(const x_t& x, bool verbose = false)
    {
        Matrix A(2, 2);
        A << 1., dt, 0., 1.;
        Vector B(2);
        B << 0.5 * dt * dt, dt;

        Matrix Q(2, 2);
        Q << 1., 0., 0., 1.;
        Scalar R = 0.1;

        Vector x0(2);
        x0 << 1., 0.;

        Vector xN(2);
        xN << 0., 0.;

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

        Scalar cost = 0.5 * x0.transpose() * Q * x0; // Initial cost (just for reference)
        for (unsigned int i = 0; i < T; i++) {
            // Action cost
            Scalar u_i = x(i);
            cost += 0.5 * R * u_i * u_i;

            grad(i) = R * u_i;
        }

        for (unsigned int i = 0; i < (T - 1); i++) {
            // State cost
            Vector x_i = x.segment(T + i * D, D);
            Vector intermediate = x_i.transpose() * (Q + Q.transpose());
            cost += 0.5 * x_i.transpose() * Q * x_i;

            for (unsigned int d = 0; d < D; d++) {
                grad(T + i * D + d) = intermediate(d);
            }
        }

        // Dynamic constraints
        for (unsigned int i = 0; i < T; i++) {
            Vector x_i = xN;
            if (i < (T - 1))
                x_i = x.segment(T + i * D, D);
            Vector x_init = x0;
            if (i > 0)
                x_init = x.segment(T + (i - 1) * D, D);
            Scalar u_i = x(i);
            Vector x_final = A * x_init + B * u_i;
            for (unsigned int d = 0; d < D; d++) {
                c(i * D + d) = x_final(d) - x_i(d);

                // Gradient wrt action
                C(i * D + d, i) = B(d);

                // Gradient wrt to current state
                if (i < (T - 1)) {
                    C(i * D + d, T + i * D + d) = -1.;
                }

                // Gradient wrt to previous state
                if (i > 0) {
                    for (unsigned int k = 0; k < D; k++) {
                        C(i * D + d, T + (i - 1) * D + k) = A(d, k);
                    }
                }
            }
        }

        Scalar cv = 0.;
        for (unsigned int i = 0; i < neq_dim; i++) {
            cv += std::abs(c[i]);
        }
        for (unsigned int i = 0; i < nin_dim; i++) {
            cv += std::abs(std::min(0., g[i]));
        }

        if (verbose) {
            std::cout << cost << " -> " << cv << std::endl;
            // std::cout << "---->" << std::endl;
            // Vector k = x0;
            // for (unsigned int i = 0; i < T - 1; i++) {
            //     std::cout << i << ": " << k.transpose() << " -> " << x(i) << std::endl;
            //     std::cout << "m: " << (A * k + B * x(i)).transpose() << std::endl;
            //     k = x.segment(T + i * D, D);
            // }
            // std::cout << (T - 1) << ": " << k.transpose() << " -> " << x(T - 1) << std::endl;
            // std::cout << "m: " << (A * k + B * x(T - 1)).transpose() << std::endl;
            // std::cout << T << ": " << xN.transpose() << std::endl;
            // std::cout << "---->" << std::endl;
        }

        return {-cost, -grad, c, C, g, G, cv};
    }
};

using DoubleInt = DoubleIntegrator<double>;

struct ParamsPSO {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = DoubleInt::dim;
    static constexpr unsigned int pop_size = 20;
    static constexpr unsigned int num_neighbors = 4;
    static constexpr unsigned int num_neighborhoods = std::floor(pop_size / static_cast<double>(num_neighbors));
    static constexpr double max_value = DoubleInt::max_value;
    static constexpr double min_value = DoubleInt::min_value;
    static constexpr double max_vel = 100.;
    static constexpr double min_vel = -100.;

    // Constraints
    static constexpr unsigned int neq_dim = DoubleInt::neq_dim;
    static constexpr unsigned int nin_dim = DoubleInt::nin_dim;

    static constexpr double chi = 0.729;
    static constexpr double c1 = 2.05;
    static constexpr double c2 = 2.05;
    static constexpr double u = 0.5;

    static constexpr bool noisy_velocity = true;
    static constexpr double mu_noise = 0.;
    static constexpr double sigma_noise = 0.0001;

    static constexpr double qp_alpha = 1.;
    static constexpr double qp_cr = 0.9;
    static constexpr double epsilon_comp = 1e-4;
};

int main()
{
    DoubleInt s;
    algevo::algo::ParticleSwarmOptimizationGrad<ParamsPSO, DoubleInt> pso;

    // // Custom Initialization
    // algevo::tools::rgen_gauss_t rgen(0., 0.001);
    // Eigen::VectorXd x0(2);
    // x0 << 1., 0.;
    // Eigen::VectorXd xN(2);
    // xN << 0., 0.;

    // for (unsigned int k = 0; k < ParamsPSO::pop_size; k++) {
    //     for (unsigned int i = 0; i < DoubleInt::T * DoubleInt::Ad; i++) {
    //         pso.population()(k, i) = 0. + rgen.rand();
    //     }

    //     for (unsigned int i = 0; i < (DoubleInt::T - 1); i++) {
    //         double p = (i + 1) / static_cast<double>(DoubleInt::T);
    //         Eigen::VectorXd x = x0 + p * (xN - x0);
    //         for (unsigned int d = 0; d < DoubleInt::D; d++) {
    //             pso.population()(k, DoubleInt::T + i * DoubleInt::D + d) = x(d) + rgen.rand();
    //         }
    //     }
    // }

    for (unsigned int i = 0; i < (1000 / ParamsPSO::pop_size); i++) {
        pso.step();
        // std::cout << i << ": " << pso.best_value() << std::endl;
        std::cout << pso.nfe() << " ";
        s.eval_all(pso.best(), true);
    }
    std::cout << pso.nfe() << " ";
    s.eval_all(pso.best(), true);
    std::cout << "Best: " << pso.best() << std::endl;
    return 0;
}
