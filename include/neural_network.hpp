#pragma once

#include <initializer_list>
#include <cassert>
#include <span>
#include <vector>
#include <iostream>
#include <algorithm>

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
        std::vector<Matrix> bias_;
        Matrix input;
        double learning_rate_;

        Matrix sigmoid(Matrix& m) {
            return m.apply([](double val) { return 1.0 / (1.0 + std::exp(-val)); });
        }

        Matrix sigmoid_derivative(Matrix& m) {
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
                layers_.push_back(Matrix::rand(*it, *(it + 1), min_rnd_num, max_rnd_num));
                bias_.push_back(Matrix::rand(1, *(it + 1), min_rnd_num, max_rnd_num));
            }
        }

        Matrix feedforward(const Matrix& input) {
            activations_.clear();
            Matrix activation = input;
            this->input = input;
            //activations_.push_back(activation);

            for (auto i = 0; i < layers_.size(); ++i) {
                activation = activation * layers_[i] + bias_[i].copy_row(0, activation.rows());
                activation = sigmoid(activation);
                activations_.push_back(activation);
            }

            return activation;
        }

        void backpropagate(const Matrix& target) {
            for (int i = 0; i < layers_.size(); ++i) {
                //std::cout << layers_[i].to_string() << std::endl;
                //std::cout << activations_[i].to_string() << std::endl;
            }

            // Compute loss derivative
            Matrix error = compute_loss_derivative(activations_.back(), target);
            Matrix delta = error.element_multiply(sigmoid_derivative(activations_.back()));
            deltas_.push_back(delta);
            
            

            for (int i = layers_.size() - 2; i >= 0; --i) {
                delta = delta * layers_[i + 1].transpose();
                //std::cout << delta.to_string() << std::endl;
                // Print delta and activation[i] sizes
                // std::cout << delta.to_string() << std::endl;
                // std::cout << activations_[i].to_string() << std::endl;
                //std::cout << "Delta size: " << delta.rows() << "x" << delta.cols() << std::endl;
                //std::cout << "Activation size: " << activations_[i].rows() << "x" << activations_[i].cols() << std::endl;
                delta = delta.element_multiply(sigmoid_derivative(activations_[i]));
                deltas_.push_back(delta);
            }

            std::reverse(deltas_.begin(), deltas_.end());
        }

        void update_weights() {
            Matrix prev_activation = this->input;

            for (std::size_t i = 0; i < layers_.size(); ++i) {
                layers_[i] = layers_[i] - ((prev_activation.transpose() * deltas_[i]).divide_scalar(deltas_[i].rows())).apply(update_weights_helper);
                prev_activation = activations_[i];
                bias_[i] = bias_[i] - deltas_[i].sum_rows().divide_scalar(deltas_[i].rows()).apply(update_weights_helper);
            }
        }

        static double update_weights_helper(double val) {
            return val * 0.2;
        }

        void fit(const Matrix& X, const Matrix& Y, std::size_t epochs) {
            for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
                Matrix output = feedforward(X);
                backpropagate(Y);
                update_weights();

                if (epoch % 1 == 0) {
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