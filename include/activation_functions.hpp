#pragma once

class ActivationFunctions {
    public:
        static double sign(double x) {
            if (x > 0.0) {
                return 1.0;
            }
            
            return -1.0;
        }
};