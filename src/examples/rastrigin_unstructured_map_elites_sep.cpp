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
#include <fstream>

#include <algevo/algo/unstructured_map_elites.hpp>
#include <algevo/tools/cvt.hpp>

#include "problems.hpp"

// Typedefs
using Rastrigin = algevo::Rastrigin2DSep<>;
using Algo = algevo::algo::MapElites<Rastrigin>;
using Params = Algo::Params;

int main()
{
    std::srand(time(0));
    // Set parameters
    Params params;
    params.dim = Rastrigin::dim;
    params.dim_features = Rastrigin::dim_features;
    params.pop_size = (params.dim > 100) ? params.dim : 128;
    params.num_cells = 20;
    params.min_dist = 0.5;
    params.max_value = Algo::x_t::Constant(params.dim, Rastrigin::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, Rastrigin::min_value);
    params.max_feat = Algo::x_t::Constant(params.dim, Rastrigin::max_features_value);
    params.min_feat = Algo::x_t::Constant(params.dim, Rastrigin::min_features_value);
    params.sigma_1 = 0.1; // 0.005;
    params.sigma_2 = 0.3;
    global::feat_collector.resize(params.pop_size, params.dim_features);

    // Instantiate algorithm
    Algo map_elites(params);

    for (unsigned int i = 0; i < 500; i++) {
        map_elites.step_evolution();
        std::cout << map_elites.params().min_dist << std::endl;
        auto log = map_elites.step_update(global::feat_collector);
        map_elites.update_min_dist(10, 0.0001, "CSC");
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << " -> archive size: " << log.archive_size << std::endl;
        const auto& archive = map_elites.all_features();
        for (unsigned int j = 0; j < log.valid_individuals.size(); j++) {
            std::cout << "    " << j << ": " << archive.col(log.valid_individuals[j]).transpose() << std::endl;
        }

        // Example of how to update the features
        if (i == 499) {
            // std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << " -> archive size: " << log.archive_size << std::endl;
            int c = 0;
            const auto& archive = map_elites.all_features();
            for (unsigned int j = 0; j < log.valid_individuals.size(); j++) {
                for (unsigned int k = 0; k < log.valid_individuals.size(); k++) {
                    double d = (archive.col(log.valid_individuals[j]) - archive.col(log.valid_individuals[k])).squaredNorm();
                    if ((k != j) && (d < params.min_dist)) {
                        c++;
                    }
                }
            }
            for (unsigned int j = 0; j < log.valid_individuals.size(); j++) {
                std::cout << "    " << j << ": " << archive.col(log.valid_individuals[j]).transpose() << std::endl;
            }
        }
    }
    return 0;
}
