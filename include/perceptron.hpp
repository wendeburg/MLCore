#include <limits.h>
#include <cassert>

#include "matrix.hpp"
#include "activation_functions.hpp"

class Perceptron {
    public:
        static Matrix train(Matrix const& X, Matrix const& Y, std::size_t const maxiter) {
            Matrix W{X.cols(), 1};
            Matrix W_best;
            int w_best_error_count = INT_MAX;

            for(auto i{0uz}; i < maxiter; i+=1) {
                Matrix H = (X * W).apply(ActivationFunctions::sign);
                Matrix Hcmp = H != Y;
                std::vector<std::size_t> error_indices = get_error_indices(Hcmp);
                int errors_commited = error_indices.size();

                if (errors_commited == 0) {
                    break;
                }

                while (!error_indices.empty()) {
                    // Si hay errores, eligo uno aleatorio y corrijo.
                    int chosen_error_index = 0;
                    if (error_indices.size() > 1) {
                        chosen_error_index = Random::get_int(0, error_indices.size()-1);
                    }

                    // w' = w + yx
                    Matrix yx = X.scalar_product(error_indices[chosen_error_index], Y[error_indices[chosen_error_index], 0]);
                    W = W + yx.transpose();
                
                    error_indices.erase(error_indices.begin() + chosen_error_index);
                }

                if (errors_commited < w_best_error_count) {
                    w_best_error_count = errors_commited;
                    W_best = W;
                }
            }

            return W_best;
        }

    private:
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

/*
Matrix train(Matrix const& X, Matrix const& Y, std::size_t const maxiter) {
   Matrix W{X.cols(), 1};

   for(auto i{0uz}; i < maxiter; i+=1) {
      Matrix H    = (X * W).apply( Matrix::sign );
      Matrix Hcmp = H != Y;
      auto err = Hcmp.sumcol(0);
   }

   // Repetir hasta maxiter iteraciones
      // Multiplicar (p.s.) todas las entradas por w (vector de pesos)
      // Comparar errores y contabilizarlos (h(X) == y)
         // Si 0 errores he terminado. Devuelvo w
         // Si hay errores, eliWo uno aleatorio y corrijo
            // w' <- w + yx
            // Si w' es mejor que wbest -> wbest = w'

   // Devolver wbest
   return X;
}
*/