#pragma once

namespace ActivationFunctions {
    using ActivationFunction = double(*)(double);
    using ActivationFunctionDerivative = double(*)(double);

    static double sign(double x) {
        if (x > 0.0) {
            return 1.0;
        }
        
        return -1.0;
    }

    static double sigmoid(double val) {
        return 1.0 / (1.0 + std::exp(-val));
    }
    
    namespace Derivatives {
        static double sigmoid_derivative(double val) { 
            return val * (1 - val); 
        }
    };

    ActivationFunctionDerivative get_derivative_from_activation(ActivationFunction f) {
        if (f == sigmoid) {
            return Derivatives::sigmoid_derivative;
        }
        else {
            return NULL;
        }
    }
};