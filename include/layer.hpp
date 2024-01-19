#include <vector>

#include "matrix.hpp"

class Layer {
    using ActivationFunction = double(*)(double);
    using ActivationFunctionDerivative = double(*)(double);

    private:
        Matrix bias_;
        Matrix weights_;
        
        Matrix activations_;
        Matrix deltas_;

        double min_rnd_num = -5;
        double max_rnd_num = 5;

        ActivationFunction activation_f;
        ActivationFunctionDerivative activation_f_deriv;

    public:
        explicit Layer(std::size_t rows, std::size_t columns) {
            weights_ = Matrix::rand(rows, columns, min_rnd_num, max_rnd_num);
            bias_ = Matrix::rand(1, columns, min_rnd_num, max_rnd_num);
            activation_f = ActivationFunctions::sigmoid;
            activation_f_deriv = ActivationFunctions::Derivatives::sigmoid_derivative;
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

        ActivationFunction activation_function() {
            return activation_f;
        }

        ActivationFunctionDerivative activation_function_derivative() {
            return activation_f_deriv;
        }
};