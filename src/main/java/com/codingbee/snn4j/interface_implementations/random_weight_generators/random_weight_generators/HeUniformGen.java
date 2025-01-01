package com.codingbee.snn4j.interface_implementations.random_weight_generators.random_weight_generators;

import com.codingbee.tool_box.exceptions.MethodCallingException;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;

import java.util.Random;

/**
 * He initialization with uniform distribution, works best in combination
 * with ReLU and leaky ReLU activation functions
 */
@SuppressWarnings("unused")
public class HeUniformGen implements RandomWeightGenerator {
    Random gen = new Random();
    @Override
    public double getWeight(int inputs, int outputs) throws MethodCallingException {
        if (inputs < 1 || outputs < 1){
            throw new MethodCallingException("The layer sizes, the weights are between, have to be larger than 0");
        }
        double limit = Math.sqrt(6.0) / Math.sqrt(inputs);
        return gen.nextDouble() * 2 * limit - limit;
    }

    @Override
    public double getHiddenLayerBias() {
        return 0;
    }

    @Override
    public double getOutputLayerBias() {
        return 0;
    }
}
