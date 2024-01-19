#pragma once

#include <limits.h>
#include <cassert>

#include "matrix.hpp"
#include "activation_functions.hpp"

class Perceptron {
    public:
        void fit(Matrix const& X, Matrix const& Y, std::size_t const maxiter) {
            Matrix new_X = X;
            new_X.add_scalar_column(1, 0);

            Matrix W{new_X.cols(), 1};

            for(auto i{0uz}; i < maxiter; i+=1) {
                Matrix H = (new_X * W).apply(ActivationFunctions::sign);
                Matrix Hcmp = H != Y;
                std::vector<std::size_t> error_indices = get_error_indices(Hcmp);
                int errors_commited = error_indices.size();

                if (errors_commited == 0) {
                    w_best_error_count = errors_commited;
                    W_best = W;

                    break;
                }

                while (!error_indices.empty()) {
                    // Si hay errores, eligo uno aleatorio y corrijo.
                    int chosen_error_index = 0;
                    if (error_indices.size() > 1) {
                        chosen_error_index = Random::get_int(0, error_indices.size()-1);
                    }

                    // w' = w + yx
                    Matrix yx = new_X.scalar_product_row(error_indices[chosen_error_index], Y[error_indices[chosen_error_index], 0]);
                    W = W + yx.transpose();
                
                    error_indices.erase(error_indices.begin() + chosen_error_index);
                }

                if (errors_commited < w_best_error_count) {
                    w_best_error_count = errors_commited;
                    W_best = W;
                }
            }
        }

        Matrix predict(Matrix const& X) {
            Matrix new_X = X;
            new_X.add_scalar_column(1, 0);

            assert(new_X.cols() == W_best.rows());

            return (new_X * W_best).apply(ActivationFunctions::sign);
        }

    private:
        Matrix W_best;
        int w_best_error_count = INT_MAX;

        static std::vector<std::size_t> get_error_indices(Matrix const &errors) {
            assert(errors.cols() == 1);
            
            std::vector<std::size_t> res;

            for (std::size_t i = 0; i < errors.rows(); i += 1) {
                if (errors[i, 0]) {
                   res.push_back(i);
                }
            }

            return res;
        }
};