#pragma once

#include <cassert>
#include <random>

class Random {
    public:
        static double get_double(double const min, double const max) {
            assert(min < max);

            static std::random_device rd{};
            static std::mt19937 gen{rd()};

            std::uniform_real_distribution<double> dist(min, max);

            return dist(gen);
        }

        static int get_int(int const min, int const max) {
            assert(min < max);

            static std::random_device rd{};
            static std::mt19937 gen{rd()};

            std::uniform_int_distribution<> dist(min, max);

            return dist(gen);
        }
};