#include <iostream>

#include <algevo/algo/pso.hpp>

struct Params {
    static constexpr unsigned int dim = 2;
    static constexpr unsigned int pop_size = 50;
    static constexpr unsigned int num_neighbors = 2;
    static constexpr unsigned int num_neighborhoods = std::ceil(pop_size / num_neighbors);
    static constexpr double max_value = 1.;
    static constexpr double min_value = -1.;
    static constexpr double max_vel = 1.;
    static constexpr double min_vel = -1.;

    static constexpr double chi = 0.729;
    static constexpr double c1 = 2.05;
    static constexpr double c2 = 2.05;
    static constexpr double u = 0.5;
};

template <typename Params, typename Scalar = double>
struct FitSphere {
    using x_t = Eigen::Matrix<Scalar, 1, Params::dim>;

    Scalar eval(const x_t& x)
    {
        return -x.squaredNorm();
    }
};

int main()
{
    algevo::algo::ParticleSwarmOptimization<Params, FitSphere<Params>> pso;

    for (unsigned int i = 0; i < 1000; i++) {
        pso.step();
        std::cout << i << ": " << pso.best() << std::endl;
    }
    return 0;
}