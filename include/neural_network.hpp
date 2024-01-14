#pragma once

#include <initializer_list>
#include <cassert>
#include <span>
#include <vector>

#include "matrix.hpp"
#include "activation_functions.hpp"

class NeuralNetwork {
    using IList = std::initializer_list<std::size_t>;

    private:
        std::vector<Matrix> layers_{};

        Matrix feedforward(const Matrix& X) {
            Matrix res{};

            res = X;

            for (auto layer : layers_) {
                res.add_scalar_column(1, 0);
                res = res * layer;
                res.apply(std::tanh);
            }

            return res;
        }

    public:
        double max_rnd_num = 10;
        double min_rnd_num = -10;

        explicit NeuralNetwork(IList arch) {
            assert(arch.size() >= 2);
            assert(*arch.begin() > 0);

            std::span arch_span {arch};
            for(auto l{1uz}; l < arch_span.size(); l+=1) {
                assert(arch_span[l] > 0);
                layers_.push_back(Matrix::rand(1 + arch_span[l-1], arch_span[l], min_rnd_num, max_rnd_num));
            }
        }

        void fit(Matrix const& X, Matrix const& Y, std::size_t const epochs) {
            for(auto i{0uz}; i < epochs; i+=1) {
                Matrix ff_result =  feedforward(X).apply(ActivationFunctions::sign);
                //auto err = 
                //   (NN.feedforward(X)
                //      .apply( Matrix::sign ) != Y
                //   )  .sumcols(0);
            }
        }   
};