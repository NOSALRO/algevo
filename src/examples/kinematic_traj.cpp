//|
//|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
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

#include <Eigen/Dense>

template <typename Scalar = double>
struct SimplePath {
    static constexpr unsigned int D = 2;
    static constexpr unsigned int T = 20;

    static constexpr unsigned int dim = T * D;
    static constexpr unsigned int dim_features = dim;
    static constexpr double max_value = 10.;
    static constexpr double min_value = -10.;
    static constexpr double max_features_value = 1.;
    static constexpr double min_features_value = 0.;
    static constexpr unsigned int neq_dim = 0;
    static constexpr unsigned int nin_dim = T + 1;

    using x_t = Eigen::Matrix<Scalar, dim, 1>;
    using c_t = Eigen::Matrix<Scalar, neq_dim, 1>;
    using g_t = Eigen::Matrix<Scalar, nin_dim, 1>;
    using C_t = Eigen::Matrix<Scalar, -1, -1>;
    using G_t = Eigen::Matrix<Scalar, -1, -1>;

    using Vector = Eigen::Matrix<Scalar, -1, 1>;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;

    std::tuple<Scalar, x_t, c_t, C_t, g_t, G_t, Scalar> eval_all(const x_t& x, bool verbose = false)
    {
        Vector x0(2);
        x0 << 10., 10.;

        Vector xN(2);
        xN << 0., 0.;

        Vector obstacle(2);
        obstacle << 5., 5.;
        Scalar obs_radius_sq = 1.;

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

        Scalar cost = 0.; // Initial cost (just for reference)
        for (unsigned int i = 0; i < T; i++) {
            Vector s_p = x0;
            if (i > 0)
                s_p = x.segment((i - 1) * D, D);
            Vector s_i = x.segment(i * D, D);
            cost += 0.5 * (s_p - s_i).squaredNorm();

            grad.segment(i * D, D) += (s_i - s_p);
            if (i > 0)
                grad.segment((i - 1) * D, D) += (s_p - s_i);
        }

        Vector x_last = x.segment((T - 1) * D, D);
        cost += 0.5 * (x_last - xN).squaredNorm();
        grad.segment((T - 1) * D, D) += (x_last - xN);

        for (unsigned int i = 0; i <= T; i++) {
            Vector s_p = x0;
            if (i > 0)
                s_p = x.segment((i - 1) * D, D);
            Vector s_i = xN;
            if (i < T)
                s_i = x.segment(i * D, D);
            Vector B = s_i; // starting point
            Vector M = s_p - s_i; // line direction
            Scalar len_M = M.dot(M) + 1e-6;
            Vector P = obstacle;
            Scalar t0 = M.dot(P - B) / len_M;
            Scalar dist;
            if (t0 <= 0)
                dist = (P - B).squaredNorm();
            else if (t0 >= 1)
                dist = (P - (B + M)).squaredNorm();
            else
                dist = (P - (B + t0 * M)).squaredNorm();
            g(i) = dist - obs_radius_sq;
            if (t0 <= 0) {
                if (i < T)
                    G.block(i, i * D, 1, D) = 2. * (B - P).transpose();
            }
            else if (t0 >= 1) {
                if (i > 0)
                    G.block(i, (i - 1) * D, 1, D) = 2. * (s_p - P).transpose();
            }
            else {
                if (i > 0)
                    G.block(i, (i - 1) * D, 1, D) = 2. * t0 * ((B + t0 * M) - P).transpose();
                if (i < T)
                    G.block(i, i * D, 1, D) = 2. * (1. - t0) * ((B + t0 * M) - P).transpose();
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
            // std::cout << x << std::endl;
            // std::cout << g << std::endl;
            // std::cout << "---->" << std::endl;
            // std::cout << G << std::endl;
            // std::cout << "::: " << G.fullPivHouseholderQr().rank() << " vs " << G.rows() << "x" << G.cols() << std::endl;
            // std::cout << grad << std::endl;
        }

        return {-cost, -grad, c, C, g, G, cv};
    }

    static void print(const x_t& sol)
    {
        std::cout << "(10, 10) -> ";
        for (unsigned int i = 0; i < T; i++) {
            std::cout << "(" << sol(i * D) << ", " << sol(i * D + 1) << ") -> ";
        }
        std::cout << "(0, 0)" << std::endl;
    }
};

// Typedefs
using SPath = SimplePath<double>;
using Algo = algevo::algo::ParticleSwarmOptimizationQP<SPath>;
using Params = Algo::Params;

int main()
{
    // Set parameters
    Params params;
    params.dim = SPath::dim;
    params.pop_size = 40;
    params.num_neighbors = 4;
    params.max_value = Algo::x_t::Constant(params.dim, SPath::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, SPath::min_value);
    params.max_vel = Algo::x_t::Constant(params.dim, 1.);
    params.min_vel = Algo::x_t::Constant(params.dim, -1.);
    params.qp_alpha = 0.5;
    params.qp_cr = 0.2;
    params.neq_dim = SPath::neq_dim;
    params.nin_dim = SPath::nin_dim;

    // Instantiate algorithm
    Algo pso(params);

    // Run a few iterations!
    Algo::IterationLog log;
    for (unsigned int i = 0; i < (5000 / params.pop_size); i++) {
        log = pso.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << " -> " << log.best_cv << std::endl;
    }

    // Print best particle
    SPath::print(log.best);

    return 0;
}
