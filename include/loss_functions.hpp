#pragma once

#include "matrix.hpp"

namespace LossFunctions {
    using LossFunction = double(*)(const Matrix&, const Matrix&);
    using LossFunctionDerivative = Matrix(*)(const Matrix&, const Matrix&);

    static double mean_squared_error(const Matrix& predictions, const Matrix& targets) {
        Matrix diff = predictions - targets;
        return diff.apply([](double val) { return val * val; }).mean();
    }
    
    namespace Derivatives {
        static Matrix mean_squared_error_derivative(const Matrix& predictions, const Matrix& targets) { 
            Matrix diff = predictions - targets;
            return diff;
        }
    };

    LossFunctionDerivative get_derivative_from_loss(LossFunction f) {
        if (f == mean_squared_error) {
            return Derivatives::mean_squared_error_derivative;
        }
        else {
            return NULL;
        }
    }
};