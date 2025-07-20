package com.codingbee.snn4j.interface_implementations.layers.ffn;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.algorithms.MemoryUtils;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.interfaces.utils.ActivationFunction;
import com.codingbee.snn4j.interfaces.utils.RandomWeightGenerator;
import com.codingbee.snn4j.interfaces.architecture.Layer;
import com.codingbee.snn4j.settings.TrainingSettings;
import java.util.Arrays;

public class FullyConnectedLayer implements Layer {
    private float[][][] weights;
    private float[][] biases;
    private int d_input;
    private int d_output;
    private int[] ds_hidden;
    private int sequenceLength;
    private TrainingSettings trainingSettings = new TrainingSettings();
    private ActivationFunction activationFunction;

    //Adam training params
    private float[][][] m_weight;
    private float[][][] v_weight;
    private float[][] m_bias;
    private float[][] v_bias;

    //Training
    //Indexed: sample, vector, layer, index
    private float[][][][] activations;
    private float[][][][] weightedSums;
    private float[][][] firstLayerInputs;

    /**
     * Create new Layer based on the specified parameters
     * @param d_input input dimensions
     * @param d_output output dimensions
     * @param ds_hidden hidden layers dimensions
     * @param sequenceLength number of vectors in the sequence
     * @param activationFunction an activation function applied to the weighted sums
     */
    public FullyConnectedLayer(int d_input, int d_output, int[] ds_hidden,
                               int sequenceLength, ActivationFunction activationFunction) {
        this.ds_hidden = ds_hidden;
        this.d_input = d_input;
        this.d_output = d_output;
        this.activationFunction = activationFunction;
        this.sequenceLength = sequenceLength;
    }

    /**
     * Empty constructor needed for Jackson
     */
    @SuppressWarnings("unused")
    public FullyConnectedLayer() {
    }

    @Override
    public float[][] process(float[][] input) {
        float[][] outputs = new float[input.length][];
        float[] layerInput;
        float[] layerOutput = null;

        for (int vector = 0; vector < input.length; vector++) {
            layerInput = input[vector];
            for (int layer = 0; layer < weights.length; layer++) {
                layerOutput = Maths.multiplyWbyV(weights[layer], layerInput);
                for (int j = 0; j < weights[layer].length; j++) {
                    layerOutput[j] = activationFunction.activate(layerOutput[j] + biases[layer][j]);
                }
                layerInput = layerOutput;
            }
            outputs[vector] = layerOutput;
        }
        return outputs;
    }

    @Override
    public float[][] forwardPass(float[][] input, int index) {
        float[][] outputs = new float[input.length][];
        float[] layerInput;
        float[] layerOutput = null;

        firstLayerInputs[index] = input;

        for (int vector = 0; vector < input.length; vector++) {
            layerInput = input[vector];
            for (int layer = 0; layer < weights.length; layer++) {
                layerOutput = new float[weights[layer].length];
                for (int j = 0; j < weights[layer].length; j++) {
                    float sum = 0;
                    for (int k = 0; k < weights[layer][j].length; k++) {
                        sum += layerInput[k] * weights[layer][j][k];
                    }
                    layerOutput[j] = sum + biases[layer][j];
                }
                weightedSums[index][vector][layer] = Arrays.copyOf(layerOutput, layerOutput.length);
                for (int k = 0; k < weights[layer].length; k++) {
                    layerOutput[k] = activationFunction.activate(layerOutput[k]);
                }
                activations[index][vector][layer] = Arrays.copyOf(layerInput, layerInput.length);

                layerInput = layerOutput;
            }
            outputs[vector] = layerOutput;
        }

        return outputs;

    }

    @Override
    public void prepareForwardPass(int numberOfSamples) {
        //Allocate an array to store inputs
        firstLayerInputs = new float[numberOfSamples][][];
        activations = new float[numberOfSamples][sequenceLength][weights.length][];
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                for (int k = 0; k < weights.length; k++) {
                    if (k == 0){
                        activations[i][j][k] = new float[d_input];
                    } else
                    {
                        activations[i][j][k] = new float[weights[k-1].length];
                    }
                }
            }
        }

        weightedSums = new float[numberOfSamples][sequenceLength][weights.length][];
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                for (int k = 0; k < weights.length; k++) {
                    weightedSums[i][j][k] = new float[weights[k].length];
                }
            }
        }
    }

    @Override
    public float[][][] backPropagateAndUpdate(float[][][] outputErrors, int adamTime) {
        if (weights == null || weights.length == 0){
            throw new IllegalCallerException("The network is not initialized");
        }
        int numberOfSamples = outputErrors.length;

        //TODO: fix for multi layer models
        float[][][] inputErrors = new float[numberOfSamples][sequenceLength][];

        float[][][][] layerErrors = new float[numberOfSamples][sequenceLength][weights.length][];

        for (int sample = 0; sample < numberOfSamples; sample++) {
            for (int vector = 0; vector < sequenceLength; vector++) {
                float[] prevLayerError = outputErrors[sample][vector];
                layerErrors[sample][vector][0] = prevLayerError;
                float[] layerError;

                for (int layer = weights.length - 2; layer >= 0; layer--){
                    layerError = Maths.multiplyVectors(Maths.multiplyTransposeWByV(weights[layer+1], prevLayerError), activationFunction.derivative(weightedSums[sample][vector][layer]));
                    layerErrors[sample][vector][layer+1] = layerError;
                    prevLayerError = layerError;
                }

            }
        }

        float[][][] weightGradient = MemoryUtils.allocateArrayOfSameSize(weights);
        float[][] biasGradient = MemoryUtils.allocateArrayOfSameSize(biases);

        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                for (int k = 0; k < weights.length; k++) {
                    float[] input;
                    if (k==0){
                        input = firstLayerInputs[i][j];
                    }else{
                        input = activations[i][j][k];
                    }
                    float[][] wg = Maths.dyadicProduct(layerErrors[i][j][(weights.length-1)-k], input);
                    Maths.addTo(weightGradient[k], wg);

                    for (int l = 0; l < weights[k].length; l++) {
                        biasGradient[k][l] += layerErrors[i][j][(weights.length-1)-k][l];
                    }
                }
            }
        }

        for (int i = 0; i < weights.length; i++) {
            Maths.scale(weightGradient[i], 1f / (numberOfSamples * sequenceLength));
        }
        Maths.scale(biasGradient, 1f / (numberOfSamples * sequenceLength));



        // Update weights and biases using calculated gradients
        updateParams(weightGradient, biasGradient, adamTime);

        // Pass to the previous layer in the model
        return inputErrors;
    }

    @Override
    public void init(RandomWeightGenerator randomGen) {
        allocateParams();
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    if (i == 0) {
                        weights[i][j][k] = randomGen.getWeight(d_input, weights[0].length);
                    } else if (i == weights.length - 1) {
                        weights[i][j][k] = randomGen.getWeight(weights[weights.length - 1].length, d_output);
                    } else {
                        weights[i][j][k] = randomGen.getWeight(weights[i].length, weights[i + 1].length);
                    }
                }
            }
        }
        for (int i = 0; i < biases.length - 1; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                biases[i][j] = randomGen.getHiddenLayerBias();
            }
        }
        for (int i = 0; i < biases[biases.length - 1].length; i++) {
            biases[biases.length - 1][i] = randomGen.getOutputLayerBias();
        }
    }

    @Override
    public void initAdamValues() {
        m_weight = MemoryUtils.allocateArrayOfSameSize(weights);
        v_weight = MemoryUtils.allocateArrayOfSameSize(weights);
        m_bias = new float[biases.length][];
        v_bias = new float[biases.length][];
        for (int i = 0; i < m_bias.length; i++) {
            m_bias[i] = new float[biases[i].length];
            v_bias[i] = new float[biases[i].length];
        }
    }

    //region Private Methods
    private void allocateParams() {
        if (ds_hidden != null && ds_hidden.length != 0) {
            weights = new float[1 + ds_hidden.length][][];
            weights[0] = new float[ds_hidden[0]][d_input];
            weights[weights.length - 1] = new float[d_output][ds_hidden[ds_hidden.length - 1]];
            for (int i = 1; i < weights.length - 1; i++) {
                weights[i] = new float[ds_hidden[i]][ds_hidden[i - 1]];
            }
            biases = new float[ds_hidden.length + 1][];
            for (int i = 0; i < biases.length - 1; i++) {
                biases[i] = new float[ds_hidden[i]];
            }
            biases[biases.length - 1] = new float[d_output];
        } else {
            weights = new float[1][d_output][d_input];
            biases = new float[1][d_output];
        }
    }

    private void updateParams(float[][][] weightGradient, float[][] biasGradient, int time) {
        //Hyperparameters
        float alpha = trainingSettings.getLearningRate();
        float beta_1 = trainingSettings.getExponentialDecayRateOne();
        float beta_2 = trainingSettings.getExponentialDecayRateTwo();
        float epsilon = trainingSettings.getEpsilon();

        //Precomputed variables to save on computations
        float oneMinusBeta_1 = 1 - beta_1;
        float oneMinusBeta_2 = 1 - beta_2;
        float oneMinusBeta_1PowTime = (float) (1 - Math.pow(beta_1, time));
        float oneMinusBeta_2PowTime = (float) (1 - Math.pow(beta_2, time));

        float g, m_hat, v_hat;

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    g = -weightGradient[i][j][k];
                    m_weight[i][j][k] = (beta_1 * m_weight[i][j][k]) + (oneMinusBeta_1 * g);
                    v_weight[i][j][k] = (beta_2 * v_weight[i][j][k]) + (oneMinusBeta_2 * g * g);
                    m_hat = m_weight[i][j][k] / oneMinusBeta_1PowTime;
                    v_hat = v_weight[i][j][k] / oneMinusBeta_2PowTime;
                    weights[i][j][k] = (float) (weights[i][j][k] - (m_hat * (alpha / (Math.sqrt(v_hat) + epsilon))));
                }
            }
        }

        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                g = -biasGradient[i][j];
                m_bias[i][j] = beta_1 * m_bias[i][j] + oneMinusBeta_1 * g;
                v_bias[i][j] = (beta_2 * v_bias[i][j] + oneMinusBeta_2 * g * g);
                m_hat = m_bias[i][j] / oneMinusBeta_1PowTime;
                v_hat = v_bias[i][j] / oneMinusBeta_2PowTime;
                biases[i][j] = (float) (biases[i][j] - m_hat * (alpha / (Math.sqrt(v_hat) + epsilon)));
            }
        }
    }
    //endregion

    //region Getter and Setters
    @Override
    public TrainingSettings getTrainingSettings() {
        return trainingSettings;
    }

    @Override
    public void setTrainingSettings(TrainingSettings trainingSettings) {
        this.trainingSettings = trainingSettings;
    }

    @Override
    public int getSequenceLength() {
        return sequenceLength;
    }

    public void setSequenceLength(int sequenceLength) {
        this.sequenceLength = sequenceLength;
    }

    @Override
    public int getInputD() {
        return d_input;
    }

    public void setInputD(int inputD) {
        d_input = inputD;
    }

    @Override
    public int getOutputD() {
        return d_output;
    }

    public void setOutputD(int outputD) {
        d_output = outputD;
    }

    public float[][][] getWeights() {
        return weights;
    }

    public void setWeights(float[][][] weights) {
        this.weights = weights;
    }

    public float[][] getBiases() {
        return biases;
    }

    public void setBiases(float[][] biases) {
        this.biases = biases;
    }

    public int getD_input() {
        return d_input;
    }

    public void setD_input(int d_input) {
        this.d_input = d_input;
    }

    public int getD_output() {
        return d_output;
    }

    public void setD_output(int d_output) {
        this.d_output = d_output;
    }

    public int[] getDs_hidden() {
        return ds_hidden;
    }

    public void setDs_hidden(int[] ds_hidden) {
        this.ds_hidden = ds_hidden;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public float[][][] getM_weight() {
        return m_weight;
    }

    public void setM_weight(float[][][] m_weight) {
        this.m_weight = m_weight;
    }

    public float[][][] getV_weight() {
        return v_weight;
    }

    public void setV_weight(float[][][] v_weight) {
        this.v_weight = v_weight;
    }

    public float[][] getM_bias() {
        return m_bias;
    }

    public void setM_bias(float[][] m_bias) {
        this.m_bias = m_bias;
    }

    public float[][] getV_bias() {
        return v_bias;
    }

    public void setV_bias(float[][] v_bias) {
        this.v_bias = v_bias;
    }
    //endregion
}