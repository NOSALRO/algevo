//|
//|    Copyright (c) 2022-2024 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2023-2024 Laboratory of Automation and Robotics, University of Patras, Greece
//|    Copyright (c) 2022-2024 Konstantinos Chatzilygeroudis
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
#ifndef ALGEVO_TOOLS_PARALLEL_HPP
#define ALGEVO_TOOLS_PARALLEL_HPP

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

#ifndef USE_TBB_ONEAPI
#include <tbb/task_scheduler_init.h>
#else
#include <oneapi/tbb/global_control.h>
using namespace oneapi;
#endif

#endif

namespace algevo {
    namespace tools {
#ifdef USE_TBB
        inline void parallel_init(int threads = -1)
        {
#ifndef USE_TBB_ONEAPI
            static tbb::task_scheduler_init init(threads);
#else
            if (threads < 0)
                threads = tbb::info::default_concurrency();
            static tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, threads);

#endif
        }
#else
        inline void parallel_init()
        {
        }
#endif

        template <typename F>
        inline void parallel_loop(size_t begin, size_t end, const F& f)
        {
#ifdef USE_TBB
            tbb::parallel_for(begin, end, static_cast<size_t>(1), [&](size_t i) {
                // clang-format off
                f(i);
                // clang-format on
            });
#else
            for (size_t i = begin; i < end; ++i)
                f(i);
#endif
        }

        template <typename Iterator, typename F>
        inline void parallel_for_each(Iterator begin, Iterator end, const F& f)
        {
#ifdef USE_TBB
            tbb::parallel_for_each(begin, end, f);
#else
            for (Iterator i = begin; i != end; ++i)
                f(*i);
#endif
        }

    } // namespace tools
} // namespace algevo

#endif
