//|
//|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
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

    // Compute centroids via CVT
    // It can take some while, if num_cells is big
    unsigned int num_points = params.num_cells * 100;
    Eigen::MatrixXd data = (Eigen::MatrixXd::Random(params.dim_features, num_points) + Eigen::MatrixXd::Constant(params.dim_features, num_points, 1.)) / 2.;
    algevo::tools::KMeans<> k(100, 1, 1e-4);
    // Set centroids
    params.centroids = k.cluster(data, params.num_cells);
    for (unsigned int c = 0; c < params.centroids.cols(); c++)
        params.centroids.col(c).array() = params.centroids.col(c).array() * (params.max_feat - params.min_feat).array() + params.min_feat.array();

    // Instantiate algorithm
    Algo map_elites(params);

    FitSphere::feat_t new_feats = FitSphere::feat_t::Zero();

    for (unsigned int i = 0; i < 2; i++) {
        auto log = map_elites.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << " -> archive size: " << log.archive_size << std::endl;
        const auto& archive = map_elites.archive_fit();
        // for (unsigned int j = 0; j < log.valid_individuals.size(); j++) {
        //     std::cout << "    " << j << ": " << archive.col(log.valid_individuals[j]).transpose() << std::endl;
        // }
        map_elites.update_features(new_feats);
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << " -> archive size: " << log.archive_size << std::endl;
    }
    return 0;
}
