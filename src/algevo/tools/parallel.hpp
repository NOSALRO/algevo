#ifndef ALGEVO_TOOLS_PARALLEL_HPP
#define ALGEVO_TOOLS_PARALLEL_HPP

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_scheduler_init.h>
#endif

namespace algevo {
    namespace tools {
#ifdef USE_TBB
        inline void parallel_init(int threads = -1)
        {
            static tbb::task_scheduler_init init(threads);
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
} // namespace ge

#endif
