package com.codingbee.snn4j.neural_networks.kan;

import com.codingbee.tool_box.exceptions.MethodCallingException;

@SuppressWarnings("unused")
public class KAN {
    //Indexed: layer, to, from, param
    private float[][][][] functionParameters;
    private float[][] biases;
    private final int inputSize;
    private final int outputSize;
    private final int hiddenLayerSize;
    private boolean initialized = false;

    public KAN(int inputSize, int outputSize, int hiddenLayerSize){
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayerSize = hiddenLayerSize;

        allocateFunctionParams();
        allocateBiases();
    }

    public float[] process(float[] input) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The KAN was not initialized with values properly yet.");
        }
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

    //region Private Methods
    private float applyFunction(float value, int layer, int to, int from){
        return (float) (Math.pow(value, 3) * functionParameters[layer][to][from][0]
                + Math.pow(value, 2) * functionParameters[layer][to][from][1]
                + value * functionParameters[layer][to][from][2]
                + functionParameters[layer][to][from][3]);
    }

    private void allocateFunctionParams(){
        functionParameters = new float[2][][][];

        functionParameters[0] = new float[hiddenLayerSize][][];
        for (int i = 0; i < hiddenLayerSize; i++) {
            functionParameters[0][i] = new float[inputSize][];
            for (int j = 0; j < inputSize; j++) {
                functionParameters[0][i][j] = new float[4];
            }
        }
        functionParameters[1] = new float[outputSize][][];
        for (int i = 0; i < outputSize; i++) {
            functionParameters[1][i] = new float[hiddenLayerSize][];
            for (int j = 0; j < hiddenLayerSize; j++) {
                functionParameters[1][i][j] = new float[4];
            }
        }
    }

    private void allocateBiases(){
        biases = new float[2][];
        biases[0] = new float[hiddenLayerSize];
        biases[1] = new float[outputSize];
    }
    //endregion Private Methods
}
