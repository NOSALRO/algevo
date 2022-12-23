#include <iostream>

#include <algevo/algo/pso.hpp>

#include "problems.hpp"

// Typedefs
using FitSphere = algevo::FitSphere<>;
using Algo = algevo::algo::ParticleSwarmOptimization<FitSphere>;
using Params = Algo::Params;

int main()
{
    // Set parameters
    Params params;
    params.dim = FitSphere::dim;
    params.pop_size = (params.dim > 100) ? params.dim : 128;
    params.num_neighbors = 10;
    params.max_value = Algo::x_t::Constant(params.dim, FitSphere::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, FitSphere::min_value);
    params.max_vel = Algo::x_t::Constant(params.dim, 1.);
    params.min_vel = Algo::x_t::Constant(params.dim, -1.);

    // Instantiate algorithm
    Algo pso(params);

    // Run a few iterations!
    for (unsigned int i = 0; i < 500; i++) {
        auto log = pso.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << std::endl;
    }

    return 0;
}
