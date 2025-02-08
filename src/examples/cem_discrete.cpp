//|
//|    Copyright (c) 2022-2025 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2023-2025 Laboratory of Automation and Robotics, University of Patras, Greece
//|    Copyright (c) 2022-2025 Konstantinos Chatzilygeroudis
//|    Authors:  Konstantinos Chatzilygeroudis
//|    email:    costashatz@gmail.com
//|    website:  https://nosalro.github.io/
//|              https://lar.upatras.gr/
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

#include <algevo/algo/cem_discrete.hpp>

template <typename Scalar = double>
struct FitDiscrete {
    static constexpr unsigned int dim = 4;
    static constexpr unsigned int num_values = 5;

    using x_t = Eigen::Matrix<unsigned int, 1, dim>;

    Scalar eval(const x_t& x)
    {
        if ((x[0] == 1 || x[0] == 2) && (x[1] == 3 || x[1] == 4) && (x[2] == 0 || x[2] == 1) && (x[3] == 4 || x[3] == 1))
            return 10.;
        return 0.;
    }
};

// Typedefs
using FitD = FitDiscrete<>;
using Algo = algevo::algo::CrossEntropyMethodDiscrete<FitD>;
using Params = Algo::Params;

int main()
{
    // Set parameters
    Params params;
    params.dim = FitD::dim;
    params.pop_size = 24;
    params.num_elites = params.pop_size * 0.8;
    params.num_values = {FitD::num_values, FitD::num_values, FitD::num_values, FitD::num_values};
    params.init_probs = Algo::p_t::Ones(FitD::dim, FitD::num_values) / static_cast<double>(FitD::num_values);

    // Instantiate algorithm
    Algo cem(params);

    // Run a few iterations!
    for (unsigned int i = 0; i < 50; i++) {
        auto log = cem.step();
        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << std::endl;
        std::cout << log.best.transpose() << std::endl;
        std::cout << cem.probabilities() << std::endl
                  << std::endl;
    }

    return 0;
}
