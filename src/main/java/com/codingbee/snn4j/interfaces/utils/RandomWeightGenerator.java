package com.codingbee.snn4j.interfaces.utils;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.MINIMAL_CLASS, property = "@class")
public interface RandomWeightGenerator {
    float getWeight(int inputs, int outputs) throws IncorrectDataException;
    float getHiddenLayerBias();
    float getOutputLayerBias();
}
