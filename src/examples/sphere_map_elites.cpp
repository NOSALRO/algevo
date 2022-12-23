#include <iostream>

#include <algevo/algo/map_elites.hpp>
#include <algevo/tools/cvt.hpp>

#include "problems.hpp"

// Typedefs
using FitSphere = algevo::FitSphere<>;
using Algo = algevo::algo::MapElites<FitSphere>;
using Params = Algo::Params;

int main()
{
    std::srand(time(0));
    // Set parameters
    Params params;
    params.dim = FitSphere::dim;
    params.dim_features = FitSphere::dim_features;
    params.pop_size = (params.dim > 100) ? params.dim : 128;
    params.num_cells = 20;
    params.max_value = Algo::x_t::Constant(params.dim, FitSphere::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, FitSphere::min_value);
    params.max_feat = Algo::x_t::Constant(params.dim, FitSphere::max_features_value);
    params.min_feat = Algo::x_t::Constant(params.dim, FitSphere::min_features_value);
    params.sigma_1 = 0.1; // 0.005;
    params.sigma_2 = 0.3;

    // Instantiate algorithm
    Algo map_elites(params);

    // Compute centroids via CVT
    // It can take some while, if num_cells is big
    unsigned int num_points = params.num_cells * 100;
    Eigen::MatrixXd data = (Eigen::MatrixXd::Random(num_points, params.dim_features) + Eigen::MatrixXd::Constant(num_points, params.dim_features, 1.)) / 2.;
    algevo::tools::KMeans<> k(100, 1, 1e-4);
    Eigen::MatrixXd centroids = k.cluster(data, params.num_cells);
    map_elites.centroids() = centroids.transpose();

    for (unsigned int i = 0; i < 500; i++) {
        auto log = map_elites.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << std::endl;
    }
    return 0;
}
