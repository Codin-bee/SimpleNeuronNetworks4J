package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.utils.ActivationFunction;

public class ELU implements ActivationFunction {
    float alpha = 1;
    @Override
    public float activate(float n) {
        if (n < 0) {
            return (float) (alpha * (Math.exp(n) - 1));
        }
        return n;
    }

    @Override
    public float derivative(float n) {
        if (n < 0) {
            return (float) (alpha * Math.exp(n));
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

    public float getAlpha() {
        return alpha;
    }

    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }
}
