package com.codingbee.snn4j.random_weight_generators;

import com.codingbee.snn4j.interfaces.RandomWeightGenerator;

import java.util.Random;

/**
 * Xavier initialization with uniform distribution, works best in combination
 * with Sigmoid and Tanh activation functions
 */
@SuppressWarnings("unused")
public class XavierUniformGen implements RandomWeightGenerator {
    Random gen = new Random();
    @Override
    public double getWeight(int inputs, int outputs) {
        double limit = Math.sqrt(6.0) / Math.sqrt(inputs + outputs);
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
