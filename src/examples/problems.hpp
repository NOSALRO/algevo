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
#ifndef ALGEVO_EXAMPLES_PROBLEMS_HPP
#define ALGEVO_EXAMPLES_PROBLEMS_HPP

#include <Eigen/Core>

namespace algevo {

    struct EvalArgs {};

    template <typename Scalar = double>
    struct FitSphere {
        static constexpr unsigned int dim = 20;
        static constexpr unsigned int dim_features = dim;
        static constexpr double max_value = 10.;
        static constexpr double min_value = -10.;
        static constexpr double max_features_value = 1.;
        static constexpr double min_features_value = 0.;

        using x_t = Eigen::Matrix<Scalar, 1, dim>;
        using feat_t = Eigen::Matrix<Scalar, 1, dim_features>;

        Scalar eval(const x_t& x)
        {
            // static constexpr double scale_range = 20.;
            // static constexpr double min_x = -10.;
            // return -x_t(x.cwiseMin(max_value).cwiseMax(min_value).array() * scale_range + min_x).squaredNorm();
            return -x.cwiseMin(max_value).cwiseMax(min_value).squaredNorm();
        }

        std::pair<Scalar, feat_t> eval_qd(const x_t& x, std::nullptr_t args) {};
        std::pair<Scalar, feat_t> eval_qd(const x_t& x)
        {
            // static constexpr double scale_range = 20.;
            // static constexpr double min_x = -10.;
            // return {-x_t(x.cwiseMin(max_value).cwiseMax(min_value).array() * scale_range + min_x).squaredNorm(), x.cwiseMin(max_features_value).cwiseMax(min_features_value)};
            return {-x.cwiseMin(max_value).cwiseMax(min_value).squaredNorm(), (x.cwiseMin(max_value).cwiseMax(min_value).array() - min_value) / (max_value - min_value)};
        }
    };

    template <typename Scalar = double>
    struct Rastrigin {
        static constexpr unsigned int dim = 2;
        static constexpr unsigned int dim_features = dim;
        static constexpr double max_value = 5.12;
        static constexpr double min_value = -5.12;
        static constexpr double max_features_value = 1.;
        static constexpr double min_features_value = 0.;

        using x_t = Eigen::Matrix<Scalar, 1, dim>;
        using feat_t = Eigen::Matrix<Scalar, 1, dim_features>;

        Scalar _value(const x_t& x)
        {
            Scalar v = static_cast<Scalar>(10. * dim);
            v += x.array().square().sum();
            v += -((static_cast<Scalar>(2. * M_PI) * x.array()).cos() * static_cast<Scalar>(10.)).sum();

            return v;
        }

        Scalar eval(const x_t& x)
        {
            return -_value(x);
        }

        std::pair<Scalar, feat_t> eval_qd(const x_t& x, std::nullptr_t args) {};
        std::pair<Scalar, feat_t> eval_qd(const x_t& x)
        {
            return {-_value(x), (x.cwiseMin(max_value).cwiseMax(min_value).array() - min_value) / (max_value - min_value)};
        }
    };
} // namespace algevo

#endif
