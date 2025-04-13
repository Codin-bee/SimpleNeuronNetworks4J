package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.utils.ActivationFunction;

@SuppressWarnings("unused")
public class Sigmoid implements ActivationFunction {
    @Override
    public float activate(float n) {
        return (float) (1 / (1 + Math.exp(-n)));
    }

    @Override
    public float derivative(float n) {
        float sig = activate(n);
        return sig * (1 - sig);
    }
}
