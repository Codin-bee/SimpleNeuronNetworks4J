package com.codingbee.snn4j.interfaces.utils;

public interface CostFunction {
    float calculate(float prediction, float target);
    float calculateAverage(float[][][] prediction, float[][][] targets);
    float calculateDerivative(float prediction, float target);
}
