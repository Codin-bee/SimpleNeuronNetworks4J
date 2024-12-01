package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.ActivationFunction;

@SuppressWarnings("unused")
public class Sigmoid implements ActivationFunction {
    @Override
    public double activate(double n) {
        return 1 / (1 + Math.exp(-n));
    }

    @Override
    public float activate(float n) {
        return (float) (1 / (1 + Math.exp(-n)));
    }
}
