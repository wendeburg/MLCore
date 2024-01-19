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
#include "train_result.hpp"

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
                // La entrada de la capa actual es la salida de la capa anterior
                z = l.get_outputs(z);
            }
        }

        void backpropagate(const Matrix& target) {
            // error = y - a^L donde y es el target y a^L es la salida de la ultima capa
            Matrix error = loss_f_derivative(layers_[layers_.size()-1].last_outputs(), target);
            // delta = error * phi'(z^L) donde phi es la funcion de perdida, z es la salida de la capa L y L es la ultima capa
            Matrix delta = error.element_multiply(layers_[layers_.size()-1].last_outputs_apply_act_func_deriv());
            // Guardamos la matriz de deltas de la ultima capa
            layers_[layers_.size()-1].set_delta(delta);

            for (int i = layers_.size()-2; i >= 0; i -= 1) {
                // Para las capas anteriores, el error es el (delta de la capa siguiente * los pesos de la capa siguiente transpuestos) 
                // multiplicado por la derivada de la funcion de activacion de la capa actual (element-wise)
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

        TrainResult fit(const Matrix& X, const Matrix& Y, std::size_t epochs, bool verbose, const Matrix& val_x = Matrix(), const Matrix& val_y = Matrix()) {
            TrainResult res;
            bool validation = !val_x.is_empty() && !val_y.is_empty();

            for (std::size_t epoch = 0; epoch < epochs; epoch += 1) {
                feedforward(X);
                backpropagate(Y);
                update_weights(X);

                double training_loss = loss_f(layers_[layers_.size()-1].last_outputs(), Y);
                res.training_loss.push_back(training_loss);

                double validation_loss = 0;
                if (validation) {
                    validation_loss = loss_f(predict(val_x), val_y);
                    res.validation_loss.push_back(validation_loss);
                }

                if (verbose) {
                        std::cout << "Epoch " << epoch << "/" << epochs << ", " << "Training Loss: " << training_loss;

                        if (validation) {
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

            return res;
        }

        Matrix predict(const Matrix& X) {
            feedforward(X);
            return layers_[layers_.size()-1].last_outputs();
        }
};