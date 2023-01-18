#include <iostream>

#include <algevo/algo/pso_qp.hpp>

template <typename Scalar = double>
struct PlanarQuad {
    static constexpr unsigned int D = 6;
    static constexpr unsigned int Ad = 2;
    static constexpr unsigned int T = 51;
    double dt = 0.05;
    double m = 1.;
    double g = 9.81;
    double l = 0.3;
    double J = 0.2 * m * l * l;

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

    Vector deriv(const Vector& x, const Vector& u)
    {
        Vector x_out(D);
        x_out.head(3) = x.tail(3);

        x_out(3) = -(u(0) + u(1)) * std::sin(x(2)) / m;
        x_out(4) = (u(0) + u(1)) * std::cos(x(2)) / m - g;
        x_out(5) = 0.5 * l / J * (u(1) - u(0));

        return x_out;
    }

    Matrix derivX(const Vector& x, const Vector& u)
    {
        Matrix d_out(D, D);
        d_out.setZero();

        d_out(0, 3) = 1.;
        d_out(1, 4) = 1.;
        d_out(2, 5) = 1.;

        d_out(3, 2) = -(u(0) + u(1)) * std::cos(x(2)) / m;
        d_out(4, 2) = -(u(0) + u(1)) * std::sin(x(2)) / m;

        d_out = d_out * dt;

        d_out(0, 0) = 1.;
        d_out(1, 1) = 1.;
        d_out(2, 2) = 1.;
        d_out(3, 3) = 1.;
        d_out(4, 4) = 1.;
        d_out(5, 5) = 1.;

        return d_out;
    }

    Matrix derivU(const Vector& x, const Vector& u)
    {
        Matrix d_out(D, Ad);
        d_out.setZero();

        d_out(3, 0) = -std::sin(x(2)) / m;
        d_out(3, 1) = -std::sin(x(2)) / m;

        d_out(4, 0) = std::cos(x(2)) / m;
        d_out(4, 1) = std::cos(x(2)) / m;

        d_out(5, 0) = -0.5 * l / J;
        d_out(5, 1) = 0.5 * l / J;

        return d_out * dt;
    }

    Vector dyn(const Vector& x, const Vector& u)
    {
        return x + deriv(x, u) * dt; // simple Euler here (might not be ideal)
    }

    std::tuple<Scalar, x_t, c_t, C_t, g_t, G_t, Scalar> eval_all(const x_t& x, bool verbose = false)
    {
        Matrix Q(D, D);
        Q.setIdentity();
        Matrix R(Ad, Ad);
        R.setIdentity();
        R *= 0.1;

        Vector x0(D);
        x0 << 1., 2., 0., 0., 0., 0.;

        Vector xN(D);
        xN << 0., 1., 0., 0., 0., 0.;

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

        Scalar cost = 0.5 * (x0 - xN).transpose() * Q * (x0 - xN); // Initial cost (just for reference)
        for (unsigned int i = 0; i < T; i++) {
            // Action cost
            Vector u_i = x.segment(i * Ad, Ad);
            cost += 0.5 * u_i.transpose() * R * u_i;

            grad.segment(i * Ad, Ad) = 0.5 * u_i.transpose() * (R + R.transpose());
        }

        for (unsigned int i = 0; i < (T - 1); i++) {
            // State cost
            Vector x_i = x.segment(T * Ad + i * D, D);
            Vector intermediate = 0.5 * x_i.transpose() * (Q + Q.transpose());
            cost += 0.5 * x_i.transpose() * Q * x_i;

            for (unsigned int d = 0; d < D; d++) {
                grad(T * Ad + i * D + d) = intermediate(d);
            }
        }

        // Dynamic constraints
        for (unsigned int i = 0; i < T; i++) {
            Vector x_i = xN;
            if (i < (T - 1))
                x_i = x.segment(T * Ad + i * D, D);
            Vector x_init = x0;
            if (i > 0)
                x_init = x.segment(T * Ad + (i - 1) * D, D);
            Vector u_i = x.segment(i * Ad, Ad);
            Vector x_final = dyn(x_init, u_i);
            Matrix dX = derivX(x_init, u_i);
            Matrix dU = derivU(x_init, u_i);
            for (unsigned int d = 0; d < D; d++) {
                c(i * D + d) = x_final(d) - x_i(d);

                // Gradient wrt action
                C.block(i * D + d, i * Ad, 1, Ad) = dU.row(d);

                // Gradient wrt to current state
                if (i < (T - 1)) {
                    C(i * D + d, T * Ad + i * D + d) = -1.;
                }

                // Gradient wrt to previous state
                if (i > 0) {
                    for (unsigned int k = 0; k < D; k++) {
                        C(i * D + d, T * Ad + (i - 1) * D + k) = dX(d, k);
                    }
                }
            }
        }

        // std::cout << x << std::endl;
        // std::cout << C << std::endl;

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

// Typedefs
using PQuad = PlanarQuad<double>;
using Algo = algevo::algo::ParticleSwarmOptimizationQP<PQuad>;
using Params = Algo::Params;

int main()
{
    // Set parameters
    Params params;
    params.dim = PQuad::dim;
    params.pop_size = 10;
    params.num_neighbors = 2;
    params.max_value = Algo::x_t::Constant(params.dim, PQuad::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, PQuad::min_value);
    params.max_vel = Algo::x_t::Constant(params.dim, 100.);
    params.min_vel = Algo::x_t::Constant(params.dim, -100.);
    params.qp_alpha = 1.;
    params.qp_cr = 0.9;
    params.neq_dim = PQuad::neq_dim;
    params.nin_dim = PQuad::nin_dim;

    // Instantiate algorithm
    Algo pso(params);

    // // Custom Initialization for faster convergence
    // algevo::tools::rgen_gauss_t rgen(0., 0.001);
    // Eigen::VectorXd x0(6);
    // x0 << 1., 2., 0., 0., 0., 0.;
    // Eigen::VectorXd xN(6);
    // xN << 0., 1., 0., 0., 0., 0.;

    // for (unsigned int k = 0; k < params.pop_size; k++) {
    //     for (unsigned int i = 0; i < PQuad::T * PQuad::Ad; i++) {
    //         pso.population()(i, k) = 0. + rgen.rand();
    //     }

    //     for (unsigned int i = 0; i < (PQuad::T - 1); i++) {
    //         double p = (i + 1) / static_cast<double>(PQuad::T);
    //         Eigen::VectorXd x = x0 + p * (xN - x0);
    //         for (unsigned int d = 0; d < PQuad::D; d++) {
    //             pso.population()(PQuad::T + i * PQuad::D + d, k) = x(d) + rgen.rand();
    //         }
    //     }
    // }

    // Run a few iterations!
    for (unsigned int i = 0; i < (1000 / params.pop_size); i++) {
        auto log = pso.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << " -> " << log.best_cv << std::endl;
    }

    // std::cout << "STD-DEV" << std::endl;
    // Eigen::VectorXd std_dev = ((pso.population().array().colwise() - pso.population().array().rowwise().mean()).square().rowwise().sum() / (params.dim - 1)).sqrt();
    // std::cout << std_dev.transpose() << std::endl;
    return 0;
}
