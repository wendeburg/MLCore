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

    // Binary step
    static double binary_step(double val) {
        if (val < 0.0) {
            return 0.0;
        }

        return 1.0;
    }

    // Linear
    static double linear(double val) {
        return val;
    }

    // Tanh
    static double tanh(double val) {
        return std::tanh(val);
    }

    // ReLU
    static double relu(double val) {
        if (val < 0.0) {
            return 0.0;
        }

        return val;
    }

    // Leaky ReLU
    static double leaky_relu(double val) {
        if (val < 0.0) {
            return 0.01 * val;
        }

        return val;
    }

    // Softmax
    static Matrix softmax(const Matrix& m) {
        assert(m.rows() == 1);

        Matrix res{m.rows(), m.cols()};
        double sum = 0.0;

        for (auto c{0uz}; c < m.cols(); c += 1) {
            sum += std::exp(m[0, c]);
        }

        for (auto c{0uz}; c < m.cols(); c += 1) {
            res[0, c] = std::exp(m[0, c]) / sum;
        }

        return res;
    }

    namespace Derivatives {
        static double sigmoid_derivative(double val) { 
            return val * (1 - val); 
        }

        static double binary_step_derivative(double val) {
            return 0.0;
        }

        static double linear_derivative(double val) {
            return 1.0;
        }

        static double tanh_derivative(double val) {
            return 1.0 - (val * val);
        }

        static double relu_derivative(double val) {
            if (val < 0.0) {
                return 0.0;
            }

            return 1.0;
        }

        static double leaky_relu_derivative(double val) {
            if (val < 0.0) {
                return 0.01;
            }

            return 1.0;
        }

        static Matrix softmax_derivative(const Matrix& m) {
            assert(m.rows() == 1);

            Matrix res{m.rows(), m.cols()};
            double sum = 0.0;

            for (auto c{0uz}; c < m.cols(); c += 1) {
                sum += std::exp(m[0, c]);
            }

            for (auto c{0uz}; c < m.cols(); c += 1) {
                res[0, c] = std::exp(m[0, c]) / sum;
            }

            return res;
        }
    };

    ActivationFunctionDerivative get_derivative_from_activation(ActivationFunction f) {
        if (f == sigmoid) {
            return Derivatives::sigmoid_derivative;
        }
        else if (f == binary_step) {
            return Derivatives::binary_step_derivative;
        }
        else if (f == linear) {
            return Derivatives::linear_derivative;
        }
        else if (f == tanh) {
            return Derivatives::tanh_derivative;
        }
        else if (f == relu) {
            return Derivatives::relu_derivative;
        }
        else if (f == leaky_relu) {
            return Derivatives::leaky_relu_derivative;
        }
        else {
            return NULL;
        }
    }
};