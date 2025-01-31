package com.codingbee.snn4j.interfaces;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

public interface RandomWeightGenerator {
    float getWeight(int inputs, int outputs) throws IncorrectDataException;
    float getHiddenLayerBias();
    float getOutputLayerBias();
}
