package com.codingbee.snn4j.interface_implementations.random_weight_generators;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.interfaces.utils.RandomWeightGenerator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class XavierGaussianGenTest {
    private final RandomWeightGenerator weightGenerator = new XavierGaussianGen();

    @Test
    void weightDistributionTest() {
        int inputs = 10;
        int outputs = 20;
        int samples = 10000;
        float mean = 0;
        float variance = 2f / (inputs + outputs);
        float sum = 0;
        float sumSq = 0;

        for (int i = 0; i < samples; i++) {
            try {
                float weight = weightGenerator.getWeight(inputs, outputs);
                sum += weight;
                sumSq += weight * weight;
            } catch (IncorrectDataException e) {
                fail("Unexpected IncorrectDataException: " + e.getMessage());
            }
        }

        float calculatedMean = sum / samples;
        float calculatedVariance = (sumSq / samples) - (calculatedMean * calculatedMean);

        assertEquals(mean, calculatedMean, 0.05, "Mean is not within expected range");
        assertEquals(variance, calculatedVariance, 0.05, "Variance is not within expected range");
    }

    @Test
    void invalidInputTest() {
        assertThrows(IncorrectDataException.class, () -> weightGenerator.getWeight(0, 10));
    }
    @Test
    void invalidInputTest2() {
        assertThrows(IncorrectDataException.class, () -> weightGenerator.getWeight(-5, 10));
    }
    @Test
    void invalidInputTest3() {
        assertThrows(IncorrectDataException.class, () -> weightGenerator.getWeight(10, 0));
    }

    @Test
    void hiddenLayerBiasTest() {
        for (int i = 0; i < 100; i++) {
            float bias = weightGenerator.getHiddenLayerBias();
            assertEquals(bias, 0);
        }
    }

    @Test
    void outputLayerBiasTest() {
        for (int i = 0; i < 100; i++) {
            float bias = weightGenerator.getOutputLayerBias();
            assertEquals(bias, 0);
        }
    }
}
