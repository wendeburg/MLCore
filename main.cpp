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

    std::vector<LayerDescriptor> arch;
    arch.emplace_back(3, ActivationFunctions::tanh);
    arch.emplace_back(1, ActivationFunctions::sigmoid);

    NeuralNetwork nn(2, arch, LossFunctions::mean_squared_error, 0.1);
    TrainResult r =  nn.fit(x_test, y_test, 10000, true);

    std::cout << "\n" << std::endl;


    std::cout << nn.predict(x_test).to_string() << std::endl;

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