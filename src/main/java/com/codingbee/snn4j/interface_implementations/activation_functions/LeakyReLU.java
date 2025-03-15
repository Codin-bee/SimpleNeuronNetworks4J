package com.codingbee.snn4j.interface_implementations.activation_functions;

import com.codingbee.snn4j.interfaces.ActivationFunction;

@SuppressWarnings("unused")
public class LeakyReLU implements ActivationFunction {
    private float alpha = 0;

    @Override
    public float activate(float n) {
        if (n < 0){
            return n * alpha;
        }else{
            return n;
        }
    }

    @Override
    public float derivative(float n) {
        if (n < 0){
            return n * alpha;
        }else{
            return alpha;
        }
    }

    public void setAlpha(float alpha){
        this.alpha = alpha;
    }

    public float getAlpha(){
        return alpha;
    }
}
