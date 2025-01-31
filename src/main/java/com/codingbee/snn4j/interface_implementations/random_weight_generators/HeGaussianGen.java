package com.codingbee.snn4j.interface_implementations.random_weight_generators;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;

import java.util.Random;

/**
 * He initialization with Gaussian(normal) distribution, works best in combination
 * with ReLU and leaky ReLU activation functions
 */
@SuppressWarnings("unused")
public class HeGaussianGen implements RandomWeightGenerator {
    Random gen = new Random();
    @Override
    public double getWeight(int inputs, int outputs) throws IncorrectDataException {
        if (inputs < 1 || outputs < 1){
            throw new IncorrectDataException("The layer sizes, the weights are between, have to be larger than 0");
        }
        return gen.nextGaussian(0, (double) 2 / inputs);
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
