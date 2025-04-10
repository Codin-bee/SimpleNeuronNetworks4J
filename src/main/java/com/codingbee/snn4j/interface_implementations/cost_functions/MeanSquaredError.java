package com.codingbee.snn4j.interface_implementations.cost_functions;

import com.codingbee.snn4j.interfaces.CostFunction;

public class MeanSquaredError implements CostFunction {
    @Override
    public float calculateAverage(float[][][] outputs, float[][][] expectedOutputs) {
        float cost = 0;
        for (int i = 0; i < outputs.length; i++) {
            for (int j = 0; j < outputs[i].length; j++) {
                for (int k = 0; k < outputs[i][j].length; k++) {
                    cost += (float) Math.pow((outputs[i][j][k] - expectedOutputs[i][j][k]), 2);
                }
            }
        }
        return cost / outputs.length;
    }
}
