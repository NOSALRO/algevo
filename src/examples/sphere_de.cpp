#include <iostream>

#include <algevo/algo/de.hpp>

#include "problems.hpp"

// Typedefs
using FitSphere = algevo::FitSphere<>;
using Algo = algevo::algo::DifferentialEvolution<FitSphere>;
using Params = Algo::Params;

int main()
{
    // Set parameters
    Params params;
    params.dim = FitSphere::dim;
    params.pop_size = (params.dim > 100) ? params.dim : 128;
    params.max_value = Algo::x_t::Constant(params.dim, FitSphere::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, FitSphere::min_value);

    // Instantiate algorithm
    Algo de(params);

    // Run a few iterations!
    for (unsigned int i = 0; i < 500; i++) {
        auto log = de.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << std::endl;
    }

    return 0;
}
