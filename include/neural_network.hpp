#pragma once

#include <initializer_list>
#include <cassert>
#include <span>
#include <vector>
#include <iostream>

#include "matrix.hpp"
#include "activation_functions.hpp"

// Loss function
double mean_squared_error(const Matrix& predictions, const Matrix& targets) {
    Matrix diff = predictions - targets;
    return diff.apply([](double val) { return val * val; }).mean();
}

class NeuralNetwork {
    using IList = std::initializer_list<std::size_t>;

    private:
        std::vector<Matrix> layers_{};
        std::vector<Matrix> activations_;
        std::vector<Matrix> deltas_;
        double learning_rate_;

        Matrix sigmoid(const Matrix& m) {
            return m.apply([](double val) { return 1.0 / (1.0 + std::exp(-val)); });
        }

        Matrix sigmoid_derivative(const Matrix& m) {
            return m.apply([](double val) { return val * (1 - val); });
        }

        Matrix compute_loss_derivative(const Matrix& predictions, const Matrix& targets) {
            return predictions - targets;
        }

    public:
        double max_rnd_num = 10;
        double min_rnd_num = -10;
        explicit NeuralNetwork(IList arch, double learning_rate = 0.01) : learning_rate_(learning_rate) {
            assert(arch.size() >= 2);
            assert(*arch.begin() > 0);
            for(auto it = arch.begin(); it != arch.end() - 1; ++it) {
                layers_.emplace_back(Matrix::rand(*it + 1, *(it + 1), min_rnd_num, max_rnd_num));
            }
        }

        Matrix feedforward(const Matrix& input) {
            activations_.clear();
            Matrix activation = input;

            for (auto& layer : layers_) {
                activation.add_scalar_column(1, 0); // Bias term
                activation = sigmoid(activation * layer);
                activations_.push_back(activation);
            }

            return activation;
        }

        void backpropagate(const Matrix& target) {
            Matrix error = compute_loss_derivative(activations_.back(), target);

            for (int i = layers_.size() - 1; i >= 0; --i) {
                Matrix delta = error * sigmoid_derivative(activations_[i]);
                deltas_.push_back(delta);

                error = delta * layers_[i].transpose();
            }

            std::reverse(deltas_.begin(), deltas_.end());
        }

        void update_weights() {
            Matrix prev_activation = activations_.front();

            for (std::size_t i = 0; i < layers_.size(); ++i) {
                layers_[i] -= (prev_activation.transpose() * deltas_[i]).apply([this](double val) { return val * learning_rate_; });
                prev_activation = activations_[i];
            }
        }

        #include <iostream>

        void fit(const Matrix& X, const Matrix& Y, std::size_t epochs) {
            for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
                Matrix output = feedforward(X);
                backpropagate(Y);
                update_weights();

                if (epoch % 100 == 0) {
                    double loss = compute_loss(output, Y);
                    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
                }
            }
        }

        double compute_loss(const Matrix& predictions, const Matrix& targets) {
            Matrix diff = predictions - targets;
            return diff.apply([](double val) { return val * val; }).mean();
        }

        Matrix predict(const Matrix& X) {
            return feedforward(X);
        }
};