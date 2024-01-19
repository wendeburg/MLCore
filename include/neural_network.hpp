#pragma once

#include <initializer_list>
#include <cassert>
#include <span>
#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>

#include "matrix.hpp"
#include "activation_functions.hpp"
#include "loss_functions.hpp"
#include "layer.hpp"
#include "layer_descriptor.hpp"

// Loss function
double mean_squared_error(const Matrix& predictions, const Matrix& targets) {
    Matrix diff = predictions - targets;
    return diff.apply([](double val) { return val * val; }).mean();
}

class NeuralNetwork {
    using ActivationFunction = ActivationFunctions::ActivationFunction;
    using LossFunction = LossFunctions::LossFunction;
    using LossFunctionDerivative = LossFunctions::LossFunctionDerivative;

    private:
        std::vector<Layer> layers_{};

        double learning_rate_;

        LossFunction loss_f;

        LossFunctionDerivative loss_f_derivative;

        Matrix compute_loss_derivative(const Matrix& predictions, const Matrix& targets) {
            return predictions - targets;
        }

    public:
        explicit NeuralNetwork(std::size_t input_dim, std::vector<LayerDescriptor> arch, LossFunction loss, double learning_rate = 0.01) : learning_rate_(learning_rate) {
            assert(arch.size() >= 1);
            assert(input_dim > 0);

            loss_f = loss;

            loss_f_derivative = LossFunctions::get_derivative_from_loss(loss);

            assert(loss_f != NULL && loss_f_derivative != NULL);

            for(auto it = arch.begin(); it != arch.end(); ++it) {
                std::size_t rows;
                ActivationFunction activf = it->activation_function();
                std::size_t columns = it->neurons_amount();

                if (it == arch.begin()) {
                    rows = input_dim;
                }
                else {
                    rows = (it-1)->neurons_amount();
                }

                layers_.emplace_back(rows, columns, activf, ActivationFunctions::get_derivative_from_activation(activf));
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
            Matrix error = loss_f_derivative(layers_[layers_.size()-1].last_outputs(), target);
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

        void fit(const Matrix& X, const Matrix& Y, std::size_t epochs, bool verbose, const Matrix& val_x = Matrix(), const Matrix& val_y = Matrix()) {
            for (std::size_t epoch = 0; epoch < epochs; epoch += 1) {
                feedforward(X);
                backpropagate(Y);
                update_weights(X);

                if (verbose) {
                    double loss = loss_f(layers_[layers_.size()-1].last_outputs(), Y);
                    std::cout << "Epoch " << epoch << "/" << epochs << ", " << "Training Loss: " << loss;

                    if (!val_x.is_empty() && !val_y.is_empty()) {
                        double validation_loss = loss_f(predict(val_x), val_y);
                        std::cout << ", Validation Loss: " << validation_loss << std::endl;
                    }
                    else {
                        std::cout << std::endl;
                    }
                }
            }

            if (verbose) {
                std::cout << "Training complete" << std::endl;
            }
        }

        Matrix predict(const Matrix& X) {
            feedforward(X);
            return layers_[layers_.size()-1].last_outputs();
        }
};