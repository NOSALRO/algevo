#include <iostream>

#include <algevo/algo/de.hpp>

struct Params {
    static constexpr int seed = -1;
    static constexpr unsigned int dim = 100;
    static constexpr unsigned int pop_size = 10 * dim;
    static constexpr double max_value = 10.;
    static constexpr double min_value = -10.;

    static constexpr double cr = 0.9;
    static constexpr double f = 0.8;
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
    algevo::algo::DifferentialEvolution<Params, FitSphere<Params>> de;

    for (unsigned int i = 0; i < 2000; i++) {
        de.step();
        std::cout << i << ": " << de.best_value() << std::endl;
    }
    return 0;
}
