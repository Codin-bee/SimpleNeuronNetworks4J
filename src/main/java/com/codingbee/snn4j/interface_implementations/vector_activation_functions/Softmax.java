package com.codingbee.snn4j.interface_implementations.vector_activation_functions;

import com.codingbee.snn4j.interfaces.utils.VectorActivationFunction;

public class Softmax implements VectorActivationFunction {

    @Override
    public float[] activate(float[] vector) {
        float max = vector[0];
        for (float v : vector) {
            if (v > max) {
                max = v;
            }
        }

        float sum = 0;
        float[] expValues = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            expValues[i] = (float) Math.exp(vector[i] - max); // for numerical stability
            sum += expValues[i];
        }

        float[] output = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            output[i] = expValues[i] / sum;
        }

        return output;
    }

    @Override
    public float[] derivative(float[] vector, float[] error) {
        float[] derivative = new float[vector.length];

        for (int i = 0; i < vector.length; i++) {
            float sum = 0f;
            for (int j = 0; j < vector.length; j++) {
                if (i == j) {
                    sum += vector[i] * (1 - vector[i]) * error[j];
                } else {
                    sum -= vector[i] * vector[j] * error[j];
                }
            }
            derivative[i] = sum;
        }

        return derivative;
    }
}
