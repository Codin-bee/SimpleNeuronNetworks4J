package com.codingbee.snn4j.activation_functions;

import com.codingbee.snn4j.interfaces.ActivationFunction;

public class ReLU implements ActivationFunction {
    @Override
    public double activate(double n) {
        if (n < 0) {
            return 0;
        }
        return n;
    }
}
