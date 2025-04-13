package com.codingbee.snn4j.interface_implementations.random_weight_generators;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.interfaces.utils.RandomWeightGenerator;

import java.util.Random;

/**
 * He initialization with uniform distribution, works best in combination
 * with ReLU and leaky ReLU activation functions
 */
@SuppressWarnings("unused")
public class HeUniformGen implements RandomWeightGenerator {
    Random gen = new Random();
    @Override
    public float getWeight(int inputs, int outputs) throws IncorrectDataException {
        if (inputs < 1 || outputs < 1){
            throw new IncorrectDataException("The layer sizes, the weights are between, have to be larger than 0");
        }
        float limit = (float) Math.sqrt(6.0 / inputs);
        return (gen.nextFloat() * 2.0f * limit) - limit;
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
