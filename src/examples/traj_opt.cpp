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
#include <iostream>

#include <algevo/algo/pso_qp.hpp>

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

// Typedefs
using DoubleInt = DoubleIntegrator<double>;
using Algo = algevo::algo::ParticleSwarmOptimizationQP<DoubleInt>;
using Params = Algo::Params;

int main()
{
    // Set parameters
    Params params;
    params.dim = DoubleInt::dim;
    params.pop_size = 20;
    params.num_neighbors = 4;
    params.max_value = Algo::x_t::Constant(params.dim, DoubleInt::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, DoubleInt::min_value);
    params.max_vel = Algo::x_t::Constant(params.dim, 100.);
    params.min_vel = Algo::x_t::Constant(params.dim, -100.);
    params.qp_alpha = 1.;
    params.qp_cr = 0.9;
    params.neq_dim = DoubleInt::neq_dim;
    params.nin_dim = DoubleInt::nin_dim;

    // Instantiate algorithm
    Algo pso(params);

    // // Custom Initialization for faster convergence
    // algevo::tools::rgen_gauss_t rgen(0., 0.001);
    // Eigen::VectorXd x0(2);
    // x0 << 1., 0.;
    // Eigen::VectorXd xN(2);
    // xN << 0., 0.;

    // for (unsigned int k = 0; k < params.pop_size; k++) {
    //     for (unsigned int i = 0; i < DoubleInt::T * DoubleInt::Ad; i++) {
    //         pso.population()(i, k) = 0. + rgen.rand();
    //     }

    //     for (unsigned int i = 0; i < (DoubleInt::T - 1); i++) {
    //         double p = (i + 1) / static_cast<double>(DoubleInt::T);
    //         Eigen::VectorXd x = x0 + p * (xN - x0);
    //         for (unsigned int d = 0; d < DoubleInt::D; d++) {
    //             pso.population()(DoubleInt::T + i * DoubleInt::D + d, k) = x(d) + rgen.rand();
    //         }
    //     }
    // }

    // Run a few iterations!
    for (unsigned int i = 0; i < (1000 / params.pop_size); i++) {
        auto log = pso.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << -log.best_value << " -> " << log.best_cv << std::endl;
    }

    return 0;
}
