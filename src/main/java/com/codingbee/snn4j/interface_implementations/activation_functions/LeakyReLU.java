package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.ActivationFunction;

@SuppressWarnings("unused")
public class LeakyReLU implements ActivationFunction {
    private double alpha = 0;

    @Override
    public double activate(double n) {
        if (n < 0){
            return n * alpha;
        }else{
            return n;
        }
    }

    @Override
    public float activate(float n) {
        if (n < 0){
            return (float) (n * alpha);
        }else{
            return n;
        }
    }

    public void setAlpha(double alpha){
        this.alpha = alpha;
    }

    public double getAlpha(){
        return alpha;
    }
}
