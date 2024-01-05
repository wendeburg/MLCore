#include "matrix.hpp"
#include "perceptron.h"

int main() {
    Matrix a = Matrix::rand(1, 2, 3, 4);
    Matrix b = Matrix::rand(1, 2, 3, 4);
    Perceptron::train(a, b, 1);
    return 0;
}