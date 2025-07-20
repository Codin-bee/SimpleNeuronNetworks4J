package com.codingbee.snn4j.interface_implementations.cost_functions;

import com.codingbee.snn4j.interfaces.utils.CostFunction;

public class MeanSquaredError implements CostFunction {
    @Override
    public float calculate(float prediction, float target) {
        float difference = prediction - target;
        return difference * difference;
    }

    @Override
    public float calculateAverage(float[][][] predictions, float[][][] targets) {
        float cost = 0;
        for (int i = 0; i < predictions.length; i++) {
            for (int j = 0; j < predictions[i].length; j++) {
                for (int k = 0; k < predictions[i][j].length; k++) {
                    cost += calculate(predictions[i][j][k], targets[i][j][k]);
                }
            }
        }
        return cost / predictions.length;
    }

    @Override
    public float calculateDerivative(float prediction, float target) {
        return -2 * (prediction - target);
    }
}
