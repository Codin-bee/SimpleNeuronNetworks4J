package com.codingbee.snn4j.interfaces;

public interface RandomWeightGenerator {
    double getWeight(int inputs, int outputs);
    double getHiddenLayerBias();
    double getOutputLayerBias();
}
