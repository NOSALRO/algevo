#include <iostream>

#include <algevo/algo/map_elites.hpp>

struct Params {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = 100;
    static constexpr unsigned int dim_features = dim;
    static constexpr unsigned int batch_size = dim;
    static constexpr double max_value = 10.;
    static constexpr double min_value = -10.;
    // static constexpr double max_value = 1.;
    // static constexpr double min_value = 0.;
    static constexpr double max_features_value = 1.;
    static constexpr double min_features_value = 0.;
    static constexpr double sigma_1 = 0.01;
    static constexpr double sigma_2 = 0.2;
    static constexpr bool grid = false;
    static constexpr unsigned int grid_size = 10;
    static constexpr unsigned int num_cells = grid ? grid_size * grid_size : 1000;
};

template <typename Params, typename Scalar = double>
struct FitSphere {
    using x_t = Eigen::Matrix<Scalar, 1, Params::dim>;
    using feat_t = Eigen::Matrix<Scalar, 1, Params::dim_features>;

    std::pair<Scalar, feat_t> eval(const x_t& x)
    {
        // static constexpr double scale_range = 20.;
        // static constexpr double min_x = -10.;
        // return {-x_t(x.array() * scale_range + min_x).squaredNorm(), x};
        return {-x.squaredNorm(), (x.array() - Params::min_features_value) / (Params::max_features_value - Params::min_features_value)};
    }
};

int main()
{
    algevo::algo::MapElites<Params, FitSphere<Params>> map_elites;

    for (unsigned int i = 0; i < 2000; i++) {
        map_elites.step();
        std::cout << i << ": " << map_elites.archive_fit().maxCoeff() << std::endl;
    }
    return 0;
}
