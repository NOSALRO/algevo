#ifndef ALGEVO_TOOLS_LIMIT_HPP
#define ALGEVO_TOOLS_LIMIT_HPP

namespace algevo {
    namespace tools {
        inline double clamp(double value, double min, double max)
        {
            return std::max(min, std::min(max, value));
        }

        inline double bounce_back(double value, double min, double max)
        {
            // TO-DO: naive implementation, see if we can make it more efficiently
            if (value < min)
                return clamp(min - value, min, max);
            else if (value > max)
                return clamp(value - max, min, max);
            return value;
        }
    } // namespace tools
} // namespace algevo

#endif
