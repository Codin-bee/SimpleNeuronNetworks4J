package com.codingbee.snn4j.interface_implementations.cost_functions;

import com.codingbee.snn4j.interfaces.utils.CostFunction;

public class CrossEntropy implements CostFunction {
    float epsilon = 1e-15f;
    @Override
    public float calculate(float prediction, float target) {
        target = Math.max(epsilon, Math.min(1 - epsilon, target));
        return (float) - (prediction * Math.log(target) + (1 - prediction) * Math.log(1 - target));
    }

    @Override
    public float calculateAverage(float[][][] prediction, float[][][] targets) {
        float sum = 0;
        for (int i = 0; i < prediction.length; i++) {
            for (int j = 0; j < prediction[i].length; j++) {
                for (int k = 0; k < prediction[i][j].length; k++) {
                    sum += calculate(prediction[i][j][k], targets[i][j][k]);
                }
            }
        }
        return sum / (prediction.length * prediction[0].length * prediction[0][0].length);
    }

    @Override
    public float calculateDerivative(float prediction, float target) {
        target = Math.max(epsilon, Math.min(1 - epsilon, target));
        return - (prediction / target - (1 - prediction) / (1 - target));
    }
}
