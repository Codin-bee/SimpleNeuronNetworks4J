package com.codingbee.snn4j.interfaces.utils;

import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.MINIMAL_CLASS, property = "@class")
public interface CostFunction {
    float calculate(float prediction, float target);
    float calculateAverage(float[][][] prediction, float[][][] targets);
    float calculateDerivative(float prediction, float target);
}
