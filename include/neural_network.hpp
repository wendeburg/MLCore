#pragma once

#include <initializer_list>
#include <cassert>
#include <span>
#include <vector>
#include <iostream>
#include <algorithm>

#include "matrix.hpp"
#include "activation_functions.hpp"
#include "layer.hpp"

// Loss function
double mean_squared_error(const Matrix& predictions, const Matrix& targets) {
    Matrix diff = predictions - targets;
    return diff.apply([](double val) { return val * val; }).mean();
}

class NeuralNetwork {
    using IList = std::initializer_list<std::size_t>;

    private:
        std::vector<Layer> layers_{};

        double learning_rate_;

        Matrix compute_loss_derivative(const Matrix& predictions, const Matrix& targets) {
            return predictions - targets;
        }

    public:
        explicit NeuralNetwork(IList arch, double learning_rate = 0.01) : learning_rate_(learning_rate) {
            assert(arch.size() >= 2);
            assert(*arch.begin() > 0);

            for(auto it = arch.begin(); it != arch.end() - 1; ++it) {
                layers_.emplace_back(*it, *(it + 1));
            }
        }

        void clear_layers_grads_and_intermediate_results() {
            for (Layer l : layers_) {
                l.zero_grads();
                l.clear_intermediate_results();
            }
        }

        void feedforward(const Matrix& input) {
            clear_layers_grads_and_intermediate_results();
            Matrix z = input;

            for (Layer &l : layers_) {
                z = l.get_outputs(z);
            }
        }

        void backpropagate(const Matrix& target) {
            Matrix error = compute_loss_derivative(layers_[layers_.size()-1].last_outputs(), target);
            Matrix delta = error.element_multiply(layers_[layers_.size()-1].last_outputs_apply_act_func_deriv());
            layers_[layers_.size()-1].set_delta(delta);

            for (int i = layers_.size()-2; i >= 0; i -= 1) {
                Matrix layer_error = layers_[i+1].deltas() * layers_[i+1].weights().transpose();
                layers_[i].set_delta(layer_error.element_multiply(layers_[i].last_outputs_apply_act_func_deriv()));
            }
        }

        void update_weights(const Matrix& input) {
            Matrix prev_outputs = input;

            for (Layer &l : layers_) {
                l.update_weights(prev_outputs, learning_rate_);
                l.update_bias(learning_rate_);
                prev_outputs = l.last_outputs();
            }
        }

        void fit(const Matrix& X, const Matrix& Y, std::size_t epochs) {
            for (std::size_t epoch = 0; epoch < epochs; epoch += 1) {
                feedforward(X);
                backpropagate(Y);
                update_weights(X);

                if (epoch % 1 == 0) {
                    double loss = compute_loss(layers_[layers_.size()-1].last_outputs(), Y);
                    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
                }
            }
        }

        double compute_loss(const Matrix& predictions, const Matrix& targets) {
            Matrix diff = predictions - targets;
            return diff.apply([](double val) { return val * val; }).mean();
        }

        Matrix predict(const Matrix& X) {
            feedforward(X);
            return layers_[layers_.size()-1].last_outputs();
        }
};