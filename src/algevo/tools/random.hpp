#ifndef ALGEVO_TOOLS_RANDOM_HPP
#define ALGEVO_TOOLS_RANDOM_HPP

#include <random>
#include <vector>

#include <algevo/external/pocketfft.h>
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
            using param_type = typename D::param_type;

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

            void param(const param_type& param) { _dist.param(param); }

        private:
            D _dist;
            std::mt19937 _rgen;
        };

        template <typename Scalar>
        class ColoredNoiseGenerator {
        public:
            using rdist_gauss_t = std::normal_distribution<Scalar>;
            using result_type = std::vector<std::vector<std::vector<Scalar>>>; // typename RandomGenerator<rdist_gauss_t>::result_type;
            using param_type = typename RandomGenerator<rdist_gauss_t>::param_type;

            using Complex = std::complex<Scalar>;

            ColoredNoiseGenerator(int seed = -1) : _normal_random(0., 1., seed) {}

            // output is (D2 x D1 x n_samples)
            result_type rand(Scalar beta, unsigned int n_samples, unsigned int D1 = 1, unsigned int D2 = 1)
            {
                const Scalar zero = static_cast<Scalar>(0.);
                const Scalar one = static_cast<Scalar>(1.);
                const Scalar two = static_cast<Scalar>(2.);
                const Scalar sqrt_2 = std::sqrt(two);

                const Scalar beta_exp = -beta / 2.;
                auto f = _rfftfreq(n_samples);
                const unsigned int n_f = f.size();

                // TO-DO: Cutoff less than f_min (we do not care about this for now)
                const Scalar f_min = one / static_cast<Scalar>(n_samples);
                unsigned int n = n_f;
                // unsigned int start = 0;
                // for (unsigned int i = 0; i < n_f; i++)
                //     if (f[i] < f_min) {
                //         n--;
                //         start = i + 1;
                //     }
                //     else
                //         break;

                unsigned int start = 0;
                for (unsigned int i = 0; i < n_f; i++)
                    if (f[i] < f_min) {
                        start = i + 1;
                    }
                    else
                        break;

                std::vector<Scalar> scales(n);

                for (unsigned int i = start; i < n; i++)
                    scales[i] = std::pow(static_cast<Scalar>(f[i]), beta_exp);
                for (unsigned int i = 0; i < start; i++)
                    scales[i] = scales[start];

                std::vector<Scalar> w(scales.begin() + 1, scales.end());
                w.back() *= (one + static_cast<Scalar>(n_samples % 2)) / two; // correct f = +-0.5

                Scalar sigma = zero;
                for (unsigned int i = 0; i < w.size(); i++)
                    sigma += w[i] * w[i];
                sigma = two * std::sqrt(sigma) / static_cast<Scalar>(n_samples);

                result_type output;
                output.resize(D2);
                for (unsigned int i = 0; i < D2; i++) {
                    output[i].resize(D1);
                    for (unsigned int j = 0; j < D1; j++) {
                        output[i][j].resize(n_samples);
                    }
                }

                for (unsigned int l = 0; l < D2; l++) {
                    for (unsigned int k = 0; k < D1; k++) {
                        std::vector<Complex> values(n_samples, Complex(zero, zero));

                        for (unsigned int i = 0; i < n; i++) {
                            if (i == 0 || ((i == (n - 1)) && ((n_samples % 2) != 0)))
                                values[i] = Complex(_normal_random.rand() * sqrt_2 * scales[i], zero);
                            else
                                values[i] = Complex(_normal_random.rand() * scales[i], _normal_random.rand() * scales[i]);
                        }

                        // std::cout << "VALUES BEFORE" << std::endl;
                        // std::cout << "===================" << std::endl;
                        // for (unsigned int i = 0; i < values.size(); i++) {
                        //     std::cout << values[i].real() << " " << values[i].imag() << "i" << std::endl;
                        // }

                        pocketfft::shape_t shape{n_samples};
                        pocketfft::stride_t s1{sizeof(Complex)};
                        pocketfft::stride_t s2{sizeof(Scalar)};
                        size_t axis = 0;
                        bool fwd = false;
                        pocketfft::c2r(shape, s1, s2, axis, fwd, values.data(), output[l][k].data(), one / (sigma * static_cast<Scalar>(n_samples)));

                        // std::cout << "VALUES AFTER" << std::endl;
                        // std::cout << "===================" << std::endl;
                        // for (unsigned int i = 0; i < output[l][k].size(); i++) {
                        //     // output[l][k][i] /= sigma;
                        //     std::cout << output[l][k][i] << std::endl;
                        // }
                        // std::cin.get();
                    }
                }

                return output;
            }

            void seed(int seed = -1)
            {
                _normal_random.seed(seed);
            }

            void reset() { _normal_random.reset(); }

            void param(const param_type& param) { _normal_random.param(param); }

        protected:
            RandomGenerator<rdist_gauss_t> _normal_random;

            std::vector<Scalar> _rfftfreq(unsigned int n, Scalar d = static_cast<Scalar>(1.)) const
            {
                Scalar val = static_cast<Scalar>(1.) / static_cast<Scalar>(n * d);

                unsigned int N = static_cast<unsigned int>(n / 2) + 1;

                std::vector<Scalar> ret(N);
                for (unsigned int i = 0; i < N; i++)
                    ret[i] = i * val;

                return ret;
            }
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
} // namespace algevo

#endif
