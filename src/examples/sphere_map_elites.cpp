#include <iostream>

#include <algevo/algo/map_elites.hpp>
#include <algevo/tools/cvt.hpp>

#include "problems.hpp"

using FitSphere = algevo::FitSphere<>;

struct Params {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = FitSphere::dim;
    static constexpr unsigned int dim_features = FitSphere::dim_features;
    static constexpr unsigned int batch_size = dim;
    static constexpr double max_value = FitSphere::max_value;
    static constexpr double min_value = FitSphere::min_value;
    static constexpr double max_features_value = FitSphere::max_features_value;
    static constexpr double min_features_value = FitSphere::min_features_value;
    // static constexpr double sigma_1 = 0.005;
    // static constexpr double sigma_2 = 0.3;
    static constexpr double sigma_1 = 0.1;
    static constexpr double sigma_2 = 0.3;
    static constexpr bool grid = false;
    static constexpr unsigned int grid_size = 10;
    static constexpr unsigned int num_cells = grid ? grid_size * grid_size : 20;
};

int main()
{
    std::srand(time(0));

    algevo::algo::MapElites<Params, FitSphere> map_elites;

    // Compute centroids via CVT
    // It can take some while, if num_cells is big
    unsigned int num_points = Params::num_cells * 100;
    Eigen::MatrixXd data = (Eigen::MatrixXd::Random(num_points, Params::dim_features) + Eigen::MatrixXd::Constant(num_points, Params::dim_features, 1.)) / 2.;
    algevo::tools::KMeans<> k(100, 1, 1e-4);
    Eigen::MatrixXd centroids = k.cluster(data, Params::num_cells);
    map_elites.centroids() = centroids;

    for (unsigned int i = 0; i < 2000; i++) {
        map_elites.step();
        std::cout << i << ": " << map_elites.archive_fit().maxCoeff() << std::endl;
    }
    return 0;
}
