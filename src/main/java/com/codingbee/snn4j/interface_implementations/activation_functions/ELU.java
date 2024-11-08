package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.ActivationFunction;

@SuppressWarnings("unused")
public class ELU implements ActivationFunction {
    double alpha = 1;
    @Override
    public double activate(double n) {
        if (n < 0) {
            return alpha * (Math.exp(n) - 1);
        }
        return n;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
}
