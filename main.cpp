#include <iostream>

#include "matrix.hpp"
#include "perceptron.hpp"
#include "neural_network.hpp"

int main() {
    // Matrix x_test({0, 0, 0,
    //                0, 0, 1,
    //                0, 1, 0,
    //                0, 1, 1,
    //                1, 0, 0,
    //                1, 0, 1,
    //                1, 1, 0,
    //                1, 1, 1}, 3);
    // Matrix y_test({0,0,0,0,0,0,0,1}, 1);
    
    // NeuralNetwork nn{3, 2, 1};
    // nn.fit(x_test, y_test, 10000);

    // Matrix x_test({0, 0,
    //                 0, 1,
    //                 1, 0,
    //                 1, 1}, 2);
    // Matrix y_test({-1,-1,-1,1}, 1);

    // Perceptron p;
    // p.fit(x_test, y_test, 1000);
    // std::cout << p.predict(x_test).to_string() << std::endl;

    Matrix x_test({0, 0,
                    0, 1,
                    1, 0,
                    1, 1}, 2);
                    
    std::cout << "Entradas: " << x_test.to_string() << std::endl;
    // Test perceptron con AND
    Matrix y_and({-1,-1,-1,1}, 1);
    Perceptron p_and;
    p_and.fit(x_test, y_and, 1000);
    std::cout << "AND: " << p_and.predict(x_test).to_string() << std::endl;

    // Test perceptron con OR
    Matrix y_or({-1,1,1,1}, 1);
    Perceptron p_or;
    p_or.fit(x_test, y_or, 1000);
    std::cout << "OR: " << p_or.predict(x_test).to_string() << std::endl;

    // Test perceptron con NAND
    Matrix y_nand({1,1,1,-1}, 1);
    Perceptron p_nand;
    p_nand.fit(x_test, y_nand, 1000);
    std::cout << "NAND: " << p_nand.predict(x_test).to_string() << std::endl;

    // Test perceptron con XOR
    Matrix y_xor({-1,1,1,-1}, 1);
    Perceptron p_xor;
    p_xor.fit(x_test, y_xor, 1000000);
    std::cout << "XOR: " << p_xor.predict(x_test).to_string() << std::endl;
    std::cout << "Iteraciones: 1000000" << std::endl;

    

    return 0;
}