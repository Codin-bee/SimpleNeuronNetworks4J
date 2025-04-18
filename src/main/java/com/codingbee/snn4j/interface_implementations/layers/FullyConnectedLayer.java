package com.codingbee.snn4j.interface_implementations.layers;

import com.codingbee.snn4j.algorithms.Maths;
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
    private float[][][][] inputs;
    private float[][][][] weightedSums;


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
                layerOutput = new float[weights[layer].length];
                for (int j = 0; j < weights[layer].length; j++) {
                    float sum = 0;
                    for (int k = 0; k < weights[layer][j].length; k++) {
                        sum += layerInput[k] * weights[layer][j][k];
                    }
                    layerOutput[j] = activationFunction.activate(sum + biases[layer][j]);
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
                inputs[index][vector][layer] = Arrays.copyOf(layerInput, layerInput.length);

                layerInput = layerOutput;
            }
            outputs[vector] = layerOutput;
        }

        return outputs;

    }

    @Override
    public void prepareForwardPass(int numberOfSamples) {
        //Allocate array to store inputs
        inputs = new float[numberOfSamples][sequenceLength][weights.length][];
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                for (int k = 0; k < weights.length; k++) {
                    if (k == 0){
                        inputs[i][j][k] = new float[d_input];
                    } else
                    {
                        inputs[i][j][k] = new float[weights[k-1].length];
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
            throw new IncorrectDataException("");
        }
        int numberOfSamples = outputErrors.length;

        //Allocate gradient arrays
        float[][][] weightGradients = Maths.allocateArrayOfSameSize(weights);
        float[][] biasGradients = new float[biases.length][];
        for (int i = 0; i < weights.length; i++) {
            biasGradients[i] = new float[biases[i].length];
        }

        //Results arrays
        float[][][] inputError = new float[numberOfSamples][sequenceLength][d_input];
        float[][][][] layerErrors = new float[numberOfSamples][sequenceLength][weights.length][];


        for (int sample = 0; sample < numberOfSamples; sample++) {
            for (int vector = 0; vector < sequenceLength; vector++) {

                float[] layerActivationDerivatives = outputErrors[sample][vector];
                float[] layerError = null;
                for (int layer = weights.length - 1; layer >= 0; layer--) {
                    float[] layerZs = weightedSums[sample][vector][layer];
                    layerError = new float[weights[layer].length];
                    for (int i = 0; i < layerError.length; i++) {
                        layerError[i] = layerActivationDerivatives[i] * activationFunction.derivative(layerZs[i]);
                    }

                    layerErrors[sample][vector][layer] = layerError;

                    //Calculating ahead for the previous layer
                    layerActivationDerivatives = Maths.multiplyTransposeWByV(weights[layer], layerError);
                }
                assert layerError != null;
                inputError[sample][vector] = Arrays.copyOf(layerError, layerError.length);

            }
        }

        //Bias gradients
        for (int layer = 0; layer < weights.length; layer++) {
            for (int index = 0; index < weights[layer].length; index++) {
                //Average results across samples and vectors
                float avgError = 0;
                for (int sample = 0; sample < numberOfSamples; sample++) {
                    for (int vector = 0; vector < sequenceLength; vector++) {
                        avgError += layerErrors[sample][vector][layer][index];
                    }
                }
                avgError /= numberOfSamples * sequenceLength;
                biasGradients[layer][index] = avgError;
                for (int i = 0; i < weights[layer][index].length; i++) {
                    weightGradients[layer][index][i] = avgError * 1;
                }
            }
        }

        //Weight gradients
        for (int layer = 0; layer < weights.length; layer++) {
            for (int index = 0; index < weights[layer].length; index++) {
                for (int connection = 0; connection < weights[layer][index].length; connection++) {
                    //Average results across samples and vectors
                    float avgGradient = 0;
                    for (int sample = 0; sample < numberOfSamples; sample++) {
                        for (int vector = 0; vector < sequenceLength; vector++) {
                            avgGradient += layerErrors[sample][vector][layer][index] * inputs[sample][vector][layer][connection];
                        }
                    }
                    avgGradient /= numberOfSamples * sequenceLength;
                    weightGradients[layer][index][connection] = avgGradient;
                }


            }
        }

        //Update weights and biases using calculate gradients
        updateParams(weightGradients, biasGradients, adamTime);

        //Pass to previous layer in the model
        return inputError;
    }


    private void updateParams(float[][][] weightGradient, float[][] biasGradient, int time) {
        //Hyperparameters
        float alpha = trainingSettings.getLearningRate();
        float beta_1 = trainingSettings.getExponentialDecayRateOne();
        float beta_2 = trainingSettings.getExponentialDecayRateTwo();
        float epsilon = trainingSettings.getEpsilon();
        float beta_3 = 1 - beta_1;
        float beta_4 = 1 - beta_2;

        float g, m_hat, v_hat;

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    g = weightGradient[i][j][k];
                    m_weight[i][j][k] = beta_1 * m_weight[i][j][k] + beta_3 * g;
                    v_weight[i][j][k] = (float) (beta_2 * v_weight[i][j][k] + beta_4 * Math.pow(g, 2));
                    m_hat = (float) (m_weight[i][j][k] / (1 - Math.pow(beta_1, time)));
                    v_hat = (float) (v_weight[i][j][k] / (1 - Math.pow(beta_2, time)));
                    weights[i][j][k] = (float) (weights[i][j][k] - m_hat * (alpha / (Math.sqrt(v_hat) + epsilon)));
                }
            }
        }

        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                g = biasGradient[i][j];
                m_bias[i][j] = beta_1 * m_bias[i][j] + beta_3 * g;
                v_bias[i][j] = (float) (beta_2 * v_bias[i][j] + beta_4 * Math.pow(g, 2));
                m_hat = (float) (m_bias[i][j] / (1 - Math.pow(beta_1, time)));
                v_hat = (float) (v_bias[i][j] / (1 - Math.pow(beta_2, time)));
                biases[i][j] = (float) (biases[i][j] - m_hat * (alpha / (Math.sqrt(v_hat) + epsilon)));
            }
        }
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
        m_weight = new float[weights.length][][];
        v_weight = new float[weights.length][][];
        for (int i = 0; i < m_weight.length; i++) {
            m_weight[i] = new float[weights[i].length][];
            v_weight[i] = new float[weights[i].length][];
            for (int j = 0; j < m_weight[i].length; j++) {
                m_weight[i][j] = new float[weights[i][j].length];
                v_weight[i][j] = new float[weights[i][j].length];
            }
        }
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

    @Override
    public void setSequenceLength(int sequenceLength) {
        this.sequenceLength = sequenceLength;
    }

    @Override
    public int getInputD() {
        return d_input;
    }

    @Override
    public void setInputD(int inputD) {
        d_input = inputD;
    }

    @Override
    public int getOutputD() {
        return d_output;
    }

    @Override
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