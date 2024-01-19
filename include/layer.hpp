#pragma once

#include <vector>

#include "matrix.hpp"
#include "activation_functions.hpp"

class Layer {
    using ActivationFunction = ActivationFunctions::ActivationFunction;
    using ActivationFunctionDerivative = ActivationFunctions::ActivationFunctionDerivative;
    
    private:
        Matrix bias_;
        Matrix weights_;
        
        Matrix activations_;
        Matrix deltas_;

        static constexpr double min_rnd_num = -0.5;
        static constexpr double max_rnd_num = 0.5;

        ActivationFunction activation_f;
        ActivationFunctionDerivative activation_f_deriv;

    public:
        explicit Layer(std::size_t rows, std::size_t columns, ActivationFunction actf, ActivationFunctionDerivative actf_deriv) {
            assert(actf != NULL && actf_deriv != NULL);

            weights_ = Matrix::rand(rows, columns, min_rnd_num, max_rnd_num);
            bias_ = Matrix::rand(1, columns, min_rnd_num, max_rnd_num);
            activation_f = actf;
            activation_f_deriv = actf_deriv;
        }

        void zero_grads() {
            deltas_.clear();
        }

        void clear_intermediate_results() {
            activations_.clear();
        }

        Matrix get_outputs(const Matrix &x) {
            activations_ = x * weights_ + (bias_.copy_row(0, x.rows()));
            activations_.apply(activation_f);
            return activations_;
        }

        Matrix last_outputs() const {
            return activations_;
        }

        Matrix last_outputs_apply_act_func_deriv() const {
            Matrix res = activations_;
            res.apply(activation_f_deriv);

            return res;
        }

        const Matrix& deltas() const {
            return deltas_;
        }

        void set_delta(const Matrix &new_deltas) {
            deltas_ = new_deltas;
        }

        const Matrix& weights() const {
            return weights_;
        }

        void update_weights(const Matrix &prev_outputs, const double &learning_rate) {
            weights_ = weights_ - (prev_outputs.transpose() * deltas_).scalar_division(deltas_.rows()).scalar_product(learning_rate);
        }

        void update_bias(const double &learning_rate) {
            bias_ = bias_ - deltas_.sum_rows().scalar_division(deltas_.rows()).scalar_product(learning_rate);
        }

        std::string to_string() const {
            std::string res = "Weights:\n";
            res += weights_.to_string();
            res += "\nBias:\n";
            res += bias_.to_string();

            return res;
        }

        ActivationFunction activation_function() {
            return activation_f;
        }

        ActivationFunctionDerivative activation_function_derivative() {
            return activation_f_deriv;
        }
};