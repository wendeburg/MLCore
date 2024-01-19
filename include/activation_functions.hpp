#pragma once

namespace ActivationFunctions {
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
};