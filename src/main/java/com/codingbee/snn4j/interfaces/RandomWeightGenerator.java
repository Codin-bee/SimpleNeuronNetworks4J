package com.codingbee.snn4j.interfaces;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

public interface RandomWeightGenerator {
    double getWeight(int inputs, int outputs) throws IncorrectDataException;
    double getHiddenLayerBias();
    double getOutputLayerBias();
}
