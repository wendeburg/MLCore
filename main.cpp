#include "matrix.hpp"
#include "perceptron.hpp"
#include "neural_network.hpp"

int main() {
    Matrix x_test({1, 3, 2, 28,
                   1, 12, 8, 10,
                   1, 9, 8, 16}, 4);
    Matrix y_test({1, -1, 1}, 1);

    Perceptron::train(x_test, y_test, 5);

    NeuralNetwork nn{2, 3, 4};
    return 0;
}