#include "matrix.hpp"

class LossFuncitons {
    public:
        static double mean_squared_error(const Matrix& predictions, const Matrix& targets) {
            Matrix diff = predictions - targets;
            return diff.apply([](double val) { return val * val; }).mean();
        }

};