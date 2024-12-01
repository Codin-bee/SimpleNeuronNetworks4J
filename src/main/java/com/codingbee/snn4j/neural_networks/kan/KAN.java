package com.codingbee.snn4j.neural_networks.kan;

public class KAN {
    private float[][][][] functionParameters;
    private float[][] biases;
    private int inputSize;
    private int outputSize;
    private int hiddenLayerSize;


    public float[] process(float[] input){
        float[] hiddenLayerNodes = new float[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                    hiddenLayerNodes[i] += applyFunction(input[j], 0, i, j);
            }
            hiddenLayerNodes[i] += biases[0][i];
        }

        float[] output = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenLayerSize; j++) {
                output[i] += applyFunction(hiddenLayerNodes[j], 1, i, j);
            }
            output[i] += biases[1][i];
        }
        return output;
    }

    private float applyFunction(float value, int layer, int to, int from){
        return (float) (Math.pow(value, 3) * functionParameters[layer][to][from][0]
                + Math.pow(value, 2) * functionParameters[layer][to][from][1]
                + value * functionParameters[layer][to][from][2]
                + functionParameters[layer][to][from][3]);
    }
}
