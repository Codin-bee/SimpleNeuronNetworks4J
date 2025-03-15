package com.codingbee.snn4j.interfaces;

public interface ActivationFunction {
    float activate(float n);
    float derivative(float n);
}
