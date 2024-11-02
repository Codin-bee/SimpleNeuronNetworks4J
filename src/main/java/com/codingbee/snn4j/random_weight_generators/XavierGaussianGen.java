package com.codingbee.snn4j.random_weight_generators;

import com.codingbee.snn4j.interfaces.RandomWeightGenerator;

import java.util.Random;

@SuppressWarnings("unused")
public class XavierGaussianGen implements RandomWeightGenerator {
    Random gen = new Random();
    @Override
    public double getWeight(int inputs, int outputs) {
        gen.nextGaussian(0, (double) 2 / (inputs + outputs));
        return 0;
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
