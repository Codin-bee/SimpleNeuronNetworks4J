package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.ActivationFunction;

@SuppressWarnings("unused")
public class Tanh implements ActivationFunction {
    @Override
    public float activate(float n) {
        return (float) ((Math.exp(n) - Math.exp(-n)) / (Math.exp(n) + Math.exp(-n)));
    }

    @Override
    public float derivative(float n) {
        float tanhVal = activate(n);
        return 1 - tanhVal * tanhVal;
    }

}
