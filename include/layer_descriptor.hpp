#include <cstddef> // std::size_t
#include <cassert>

#include "activation_functions.hpp"

class LayerDescriptor {
    using ActivationFunction = ActivationFunctions::ActivationFunction;

    private:
        std::size_t neurons;
        ActivationFunction activation_f;

    public:
        explicit LayerDescriptor(std::size_t neurons, ActivationFunction activation_f) {
            assert(neurons > 0);

            this->neurons = neurons;
            this->activation_f = activation_f;
        }

        std::size_t neurons_amount() const {
            return neurons;
        }

        ActivationFunction activation_function() const {
            return activation_f;
        }
};