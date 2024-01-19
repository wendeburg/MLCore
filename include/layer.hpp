#include <vector>

#include "matrix.hpp"

class Layer {
    private:
        Matrix bias_;
        Matrix weights_;
        
        Matrix activations_;
        Matrix deltas_;

        double min_rnd_num = -5;
        double max_rnd_num = 5;

    public:
        explicit Layer(std::size_t rows, std::size_t columns) {
            weights_ = Matrix::rand(rows, columns, min_rnd_num, max_rnd_num);
            bias_ = Matrix::rand(1, columns, min_rnd_num, max_rnd_num);
        }

        void zero_grads() {
            deltas_.clear();
        }

        void clear_intermediate_results() {
            activations_.clear();
        }

        Matrix get_outputs(const Matrix &x) {
            activations_ = x * weights_ + (bias_.copy_row(0, x.rows()));
            activations_.apply(ActivationFunctions::sigmoid);
            return activations_;
        }

        Matrix last_outputs() const {
            return activations_;
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
};