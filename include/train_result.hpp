#pragma once

#include <vector>

class TrainResult {
    public:
        std::vector<double> training_loss;
        std::vector<double> validation_loss;
};