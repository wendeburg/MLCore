#include <iostream>

#include "matrix.hpp"
#include "perceptron.hpp"
#include "neural_network.hpp"

int main() {
    Matrix x_test({0,0,
                   0,1,
                   1,0,
                   1,1,}, 2);
    Matrix y_test({0,1,1,0}, 1);
    
    NeuralNetwork nn{2, 2, 1};
    nn.fit(x_test, y_test, 100);

    // Matrix x_test({0, 0,
    //                 0, 1,
    //                 1, 0,
    //                 1, 1}, 2);
    // Matrix y_test({-1,-1,-1,1}, 1);

    // Perceptron p;
    // p.fit(x_test, y_test, 1000);
    // std::cout << p.predict(x_test).to_string() << std::endl;

    return 0;
}