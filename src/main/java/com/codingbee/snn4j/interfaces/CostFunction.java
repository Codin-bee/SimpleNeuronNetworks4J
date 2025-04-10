package com.codingbee.snn4j.interfaces;

public interface CostFunction {
    float calculateAverage(float[][][] outputs, float[][][] expectedOutputs);
}
