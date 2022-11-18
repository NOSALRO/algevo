#ifndef ALGEVO_EXAMPLES_PROBLEMS_HPP
#define ALGEVO_EXAMPLES_PROBLEMS_HPP

#include <Eigen/Core>

namespace algevo {
    template <typename Scalar = double>
    struct FitSphere {
        static constexpr unsigned int dim = 100;
        static constexpr unsigned int dim_features = dim;
        static constexpr double max_value = 10.;
        static constexpr double min_value = -10.;
        static constexpr double max_features_value = 1.;
        static constexpr double min_features_value = 0.;

        using x_t = Eigen::Matrix<Scalar, 1, dim>;
        using feat_t = Eigen::Matrix<Scalar, 1, dim_features>;

        Scalar eval(const x_t& x)
        {
            // static constexpr double scale_range = 20.;
            // static constexpr double min_x = -10.;
            // return -x_t(x.cwiseMin(max_value).cwiseMax(min_value).array() * scale_range + min_x).squaredNorm();
            return -x.cwiseMin(max_value).cwiseMax(min_value).squaredNorm();
        }

        std::pair<Scalar, feat_t> eval_qd(const x_t& x)
        {
            // static constexpr double scale_range = 20.;
            // static constexpr double min_x = -10.;
            // return {-x_t(x.cwiseMin(max_value).cwiseMax(min_value).array() * scale_range + min_x).squaredNorm(), x.cwiseMin(max_features_value).cwiseMax(min_features_value)};
            return {-x.cwiseMin(max_value).cwiseMax(min_value).squaredNorm(), (x.cwiseMin(max_value).cwiseMax(min_value).array() - min_value) / (max_value - min_value)};
        }
    };
} // namespace algevo

#endif
