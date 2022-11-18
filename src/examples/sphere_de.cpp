#include <iostream>

#include <algevo/algo/de.hpp>

#include "problems.hpp"

using FitSphere = algevo::FitSphere<>;

struct Params {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = FitSphere::dim;
    static constexpr unsigned int pop_size = (dim > 100) ? dim : 128;
    static constexpr double max_value = FitSphere::max_value;
    static constexpr double min_value = FitSphere::min_value;

    static constexpr double cr = 0.9;
    static constexpr double f = 0.8;
    static constexpr double lambda = f;
};

int main()
{
    algevo::algo::DifferentialEvolution<Params, FitSphere> de;

    for (unsigned int i = 0; i < 2000; i++) {
        de.step();
        std::cout << i << ": " << de.best_value() << std::endl;
    }
    return 0;
}
