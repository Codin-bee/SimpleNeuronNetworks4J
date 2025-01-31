package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.ActivationFunction;

@SuppressWarnings("unused")
public class ELU implements ActivationFunction {
    float alpha = 1;
    @Override
    public float activate(float n) {
        if (n < 0) {
            return (float) (alpha * (Math.exp(n) - 1));
        }
        return n;
    }

    public float getAlpha() {
        return alpha;
    }

    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }
}
