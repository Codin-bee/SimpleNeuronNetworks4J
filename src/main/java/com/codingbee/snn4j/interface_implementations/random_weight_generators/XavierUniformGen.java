package com.codingbee.snn4j.interface_implementations.random_weight_generators;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.interfaces.utils.RandomWeightGenerator;

import java.util.Random;

/**
 * Xavier initialization with uniform distribution, works best in combination
 * with Sigmoid and Tanh activation functions
 */
public class XavierUniformGen implements RandomWeightGenerator {
    Random gen = new Random();
    @Override
    public float getWeight(int inputs, int outputs) throws IncorrectDataException {
        if (inputs < 1 || outputs < 1){
            throw new IncorrectDataException("The layer sizes, the weights are between, have to be larger than 0");
        }
        double limit = Math.sqrt(6.0) / Math.sqrt(inputs + outputs);
        return (float) (gen.nextDouble() * 2 * limit - limit);
    }

    @Override
    public float getHiddenLayerBias() {
        return 0;
    }

    @Override
    public float getOutputLayerBias() {
        return 0;
    }
}
