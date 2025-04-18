package com.codingbee.snn4j.interfaces.utils;

public interface VectorActivationFunction {
    float[] activate(float [] vector);
    float[] derivative(float[] vector);
}
