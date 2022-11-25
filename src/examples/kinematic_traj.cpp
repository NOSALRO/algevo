#include <iostream>

#include <algevo/algo/pso_grad.hpp>

#include <Eigen/Dense>

template <typename Scalar = double>
struct SimplePath {
    static constexpr unsigned int D = 2;
    static constexpr unsigned int T = 10;

    static constexpr unsigned int dim = T * D;
    static constexpr unsigned int dim_features = dim;
    static constexpr double max_value = 10.;
    static constexpr double min_value = -10.;
    static constexpr double max_features_value = 1.;
    static constexpr double min_features_value = 0.;
    // static constexpr unsigned int neq_dim = 1;
    // static constexpr unsigned int nin_dim = 0;
    static constexpr unsigned int neq_dim = 0;
    static constexpr unsigned int nin_dim = T + 1;

    using x_t = Eigen::Matrix<Scalar, 1, dim>;
    using c_t = Eigen::Matrix<Scalar, 1, neq_dim>;
    using g_t = Eigen::Matrix<Scalar, 1, nin_dim>;
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
};

using SPath = SimplePath<double>;

struct ParamsPSO {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = SPath::dim;
    static constexpr unsigned int pop_size = 40;
    static constexpr unsigned int num_neighbors = 4;
    static constexpr unsigned int num_neighborhoods = std::floor(pop_size / static_cast<double>(num_neighbors));
    static constexpr double max_value = SPath::max_value;
    static constexpr double min_value = SPath::min_value;
    static constexpr double max_vel = 1.;
    static constexpr double min_vel = -1.;

    // Constraints
    static constexpr unsigned int neq_dim = SPath::neq_dim;
    static constexpr unsigned int nin_dim = SPath::nin_dim;

    static constexpr double chi = 0.729;
    static constexpr double c1 = 2.05;
    static constexpr double c2 = 2.05;
    static constexpr double u = 0.5;

    static constexpr bool noisy_velocity = true;
    static constexpr double mu_noise = 0.;
    static constexpr double sigma_noise = 0.0001;

    static constexpr double qp_alpha = 1.;
    static constexpr double qp_cr = 0.2;
    static constexpr double epsilon_comp = 1e-4;
};

int main()
{
    SPath s;
    algevo::algo::ParticleSwarmOptimizationGrad<ParamsPSO, SPath> pso;

    for (unsigned int i = 0; i < 100; i++) {
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
