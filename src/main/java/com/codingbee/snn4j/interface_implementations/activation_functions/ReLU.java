package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.utils.ActivationFunction;

@SuppressWarnings("unused")
public class ReLU implements ActivationFunction {
    @Override
    public float activate(float n) {
        if (n < 0) {
            return 0;
        }
        return n;
    }
    @Override
    public float derivative(float n) {
        if (n < 0) {
            return 0;
        }
        return 1;
    }

    @Override
    public float[] derivative(float[] ns) {
        float[] derivatives = new float[ns.length];
        for (int i = 0; i < ns.length; i++) {
            derivatives[i] = derivative(ns[i]);
        }
        return derivatives;
    }
}
