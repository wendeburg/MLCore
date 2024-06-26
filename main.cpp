#include <iostream>
#include <fstream>
#include <sstream>

#include "matrix.hpp"
#include "perceptron.hpp"
#include "neural_network.hpp"

int main() {
    // ==================== NEURAL NETWORK tests ====================

    // ==================== XOR ====================
    // Matrix x_test({0,0,
    //                0,1,
    //                1,0,
    //                1,1,}, 2);
    // Matrix y_test({0,1,1,0}, 1);

    // std::vector<LayerDescriptor> arch;
    // arch.emplace_back(2, ActivationFunctions::tanh);
    // arch.emplace_back(1, ActivationFunctions::tanh);

    // NeuralNetwork nn(2, arch, LossFunctions::mean_squared_error, 0.1);
    // TrainResult r =  nn.fit(x_test, y_test, 10000, true);

    // std::cout << "\n" << std::endl;
    // std::cout << nn.predict(x_test).to_string() << std::endl;


    // ==================== Abalone ====================

    // Import abalone dataset from csv
    // Matrix x_test({
    //     #include "abalone.csv"
    //     }, 9);
    // Matrix y_test({
    //     #include "abalone_target.csv"
    //     }, 2);
    
    // std::vector<LayerDescriptor> arch;
    // arch.emplace_back(32, ActivationFunctions::tanh);
    // arch.emplace_back(1, ActivationFunctions::tanh);

    // NeuralNetwork nn(8, arch, LossFunctions::mean_squared_error, 0.1);
    // TrainResult r =  nn.fit(x_test.remove_column(0), y_test.remove_column(0), 10000, true);


    // ==================== Housing ====================

    // Housing dataset
    // Matrix x_train({
    //     #include "housing.csv"
    //     }, 14);
    // Matrix y_train = x_train.get_col(13);

    // Matrix x_val({
    //     #include "housing_validation.csv"
    //     }, 14);
    // Matrix y_val = x_val.get_col(13);
    
    // std::vector<LayerDescriptor> arch;
    // arch.emplace_back(32, ActivationFunctions::sigmoid);
    // arch.emplace_back(1, ActivationFunctions::linear);

    // NeuralNetwork nn(13, arch, LossFunctions::mean_squared_error, 0.0001);
    // TrainResult r =  nn.fit(x_train.remove_column(0), y_train, 100000, true, x_val.remove_column(0), y_val, 500);



    // ==================== Sin ====================

    // Test with sin function
    // Matrix x_train({
    //     #include "x_train.csv"
    //     }, 1);
    // Matrix y_train({
    //     #include "y_train.csv"
    //     }, 1);
    // Matrix x_val({
    //     #include "x_test.csv"
    //     }, 1);
    // Matrix y_val({
    //     #include "y_test.csv"
    //     }, 1);
    
    // std::vector<LayerDescriptor> arch;
    // arch.emplace_back(8, ActivationFunctions::tanh);
    // arch.emplace_back(1, ActivationFunctions::tanh);

    // NeuralNetwork nn(1, arch, LossFunctions::mean_squared_error, 0.5);
    // TrainResult r =  nn.fit(x_train, y_train, 100000, true, x_val, y_val);

    // // Compare with actual sin function
    // Matrix x_show({
    //     0, 0.1, 0.2, 0.3, 0.4, 0.5,
    //     }, 1);
    // std::cout << "Predicted: " << nn.predict(x_show).to_string() << std::endl;
    // std::cout << "Actual: " << x_show.apply([](double val) { return std::sin(val); }).to_string() << std::endl;



    // ==================== SinC ====================


    // Test with sinC function
    // Matrix x_train({
    //     #include "xsinc_train.csv"
    //     }, 1);
    // Matrix y_train({
    //     #include "ysinc_train.csv"
    //     }, 1);
    // Matrix x_val({
    //     #include "xsinc_test.csv"
    //     }, 1);
    // Matrix y_val({
    //     #include "ysinc_test.csv"
    //     }, 1);

    // std::vector<LayerDescriptor> arch;
    // arch.emplace_back(8, ActivationFunctions::tanh);
    // arch.emplace_back(1, ActivationFunctions::tanh);

    // NeuralNetwork nn(1, arch, LossFunctions::mean_squared_error, 0.1);
    // TrainResult r =  nn.fit(x_train, y_train, 10000, true, x_val, y_val);

    // // Compare with actual sinc function
    // Matrix x_show({
    //     0, -3.9, -2.1, 3.5, 4.2, 2.3,
    //     }, 1);
    // std::cout << "Predicted: " << nn.predict(x_show).to_string() << std::endl;
    // std::cout << "Actual: " << x_show.apply([](double val) { return std::sin(M_PI*val) / (M_PI*val); }).to_string() << std::endl;


    // ==================== PERCEPTRON tests ====================


    // Matrix x_test({0, 0,
    //                 0, 1,
    //                 1, 0,
    //                 1, 1}, 2);
                    
    // std::cout << "Entradas: " << x_test.to_string() << std::endl;



    // ==================== AND ====================

    // Matrix y_and({-1,-1,-1,1}, 1);
    // Perceptron p_and;
    // p_and.fit(x_test, y_and, 1000);
    // std::cout << "AND: " << p_and.predict(x_test).to_string() << std::endl;

    
    
    // ==================== OR ====================

    // Matrix y_or({-1,1,1,1}, 1);
    // Perceptron p_or;
    // p_or.fit(x_test, y_or, 1000);
    // std::cout << "OR: " << p_or.predict(x_test).to_string() << std::endl;

    
    
    // ==================== NAND ====================

    // Matrix y_nand({1,1,1,-1}, 1);
    // Perceptron p_nand;
    // p_nand.fit(x_test, y_nand, 1000);
    // std::cout << "NAND: " << p_nand.predict(x_test).to_string() << std::endl;

    
    
    // ==================== NOR ====================

    // Matrix y_xor({-1,1,1,-1}, 1);
    // Perceptron p_xor;
    // p_xor.fit(x_test, y_xor, 1000000);
    // std::cout << "XOR: " << p_xor.predict(x_test).to_string() << std::endl;
    // std::cout << "Iteraciones: 1000000" << std::endl;

    return 0;
}