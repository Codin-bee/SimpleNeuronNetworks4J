package com.codingbee.snn4j.interfaces;

import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.MINIMAL_CLASS, property = "@class")
public interface ActivationFunction {
    float activate(float n);
    float derivative(float n);
}
