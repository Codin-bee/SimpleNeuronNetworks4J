package com.codingbee.snn4j.interfaces.utils;

import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.MINIMAL_CLASS, property = "@class")
public interface VectorActivationFunction {
    float[] activate(float [] vector);
    float[] derivative(float[] vector, float[] error);
}
