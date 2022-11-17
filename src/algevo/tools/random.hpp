#ifndef ALGEVO_TOOLS_RANDOM_HPP
#define ALGEVO_TOOLS_RANDOM_HPP

#include <random>

#include <algevo/external/rand_utils.hpp>

namespace algevo {
    namespace tools {
        /// a mt19937-based random generator (mutex-protected)
        ///
        /// usage :
        /// - RandomGenerator<dist<double>>(0.0, 1.0);
        /// - double r = rgen.rand();
        template <typename D>
        class RandomGenerator {
        public:
            using result_type = typename D::result_type;
            RandomGenerator(result_type a, result_type b, int seed = -1) : _dist(a, b) { this->seed(seed); }

            result_type rand() { return _dist(_rgen); }

            void seed(int seed = -1)
            {
                if (seed >= 0)
                    _rgen.seed(seed);
                else
                    _rgen.seed(randutils::auto_seed_128{}.base());
            }

            void reset() { _dist.reset(); }

            void param(const typename D::param_type& param) { _dist.param(param); }

        private:
            D _dist;
            std::mt19937 _rgen;
        };

        using rdist_double_t = std::uniform_real_distribution<double>;

        using rdist_int_t = std::uniform_int_distribution<int>;

        using rdist_gauss_t = std::normal_distribution<>;

        /// Double random number generator
        using rgen_double_t = RandomGenerator<rdist_double_t>;

        /// Double random number generator (gaussian)
        using rgen_gauss_t = RandomGenerator<rdist_gauss_t>;

        /// integer random number generator
        using rgen_int_t = RandomGenerator<rdist_int_t>;

    } // namespace tools
} // namespace ge

#endif
