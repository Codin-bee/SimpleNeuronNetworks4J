package com.codingbee.snn4j.interface_implementations.random_weight_generators.random_weight_generators;

import com.codingbee.snn4j.exceptions.MethodCallingException;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;

import java.util.Random;

/**
 * Xavier initialization with Gaussian(normal) distribution, works best in combination
 * with Sigmoid and Tanh activation functions
 */
@SuppressWarnings("unused")
public class XavierGaussianGen implements RandomWeightGenerator {
    Random gen = new Random();
    @Override
    public double getWeight(int inputs, int outputs) throws MethodCallingException {
        if (inputs < 1 || outputs < 1){
            throw new MethodCallingException("The layer sizes, the weights are between, have to be larger than 0");
        }
        return gen.nextGaussian(0, (double) 2 / (inputs + outputs));
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
