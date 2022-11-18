#include <iostream>

#include <algevo/algo/pso.hpp>

#include "problems.hpp"

using FitSphere = algevo::FitSphere<>;

struct Params {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = FitSphere::dim;
    static constexpr unsigned int pop_size = 100;
    static constexpr unsigned int num_neighbors = 10;
    static constexpr unsigned int num_neighborhoods = std::ceil(pop_size / num_neighbors);
    static constexpr double max_value = FitSphere::max_value;
    static constexpr double min_value = FitSphere::min_value;
    static constexpr double max_vel = 1.;
    static constexpr double min_vel = -1.;

    static constexpr double chi = 0.729;
    static constexpr double c1 = 2.05;
    static constexpr double c2 = 2.05;
    static constexpr double u = 0.5;

    static constexpr bool noisy_velocity = true;
    static constexpr double mu_noise = 0.;
    static constexpr double sigma_noise = 0.0001;
};

int main()
{
    algevo::algo::ParticleSwarmOptimization<Params, FitSphere> pso;

    for (unsigned int i = 0; i < 2000; i++) {
        pso.step();
        std::cout << i << ": " << pso.best_value() << std::endl;
    }
    return 0;
}
