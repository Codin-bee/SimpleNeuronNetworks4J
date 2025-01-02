package com.codingbee.snn4j.interfaces;

import com.codingbee.tool_box.exceptions.MethodCallingException;

public interface RandomWeightGenerator {
    double getWeight(int inputs, int outputs) throws MethodCallingException;
    double getHiddenLayerBias();
    double getOutputLayerBias();
}
