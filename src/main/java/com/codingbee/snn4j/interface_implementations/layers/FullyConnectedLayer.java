package com.codingbee.snn4j.interface_implementations.layers;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.ActivationFunction;
import com.codingbee.snn4j.interfaces.model.RandomWeightGenerator;
import com.codingbee.snn4j.interfaces.model.Layer;
import com.codingbee.snn4j.interfaces.model.Model;
import com.codingbee.snn4j.settings.TrainingSettings;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class FullyConnectedLayer implements Layer {
    private float[][][] weights;
    private float[][] biases;
    private int inputLayerSize;
    private int outputLayerSize;
    private int[] hiddenLayersSizes;
    private TrainingSettings trainingSettings = new TrainingSettings();
    private ActivationFunction activationFunction;
    private Model fullModel;

    //Adam optimizer values
    float alpha;
    float beta_1;
    float beta_2;
    float beta_3;
    float beta_4;
    float epsilon;
    float time;

    float[][][] m_weight;
    float[][][] v_weight;
    float[][] m_bias;
    float[][] v_bias;


    public FullyConnectedLayer(int inputLayerSize, int outputLayerSize,int[] hiddenLayersSizes,
                               ActivationFunction activationFunction) {
        this.hiddenLayersSizes = hiddenLayersSizes;
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.activationFunction = activationFunction;

        allocateParams();
    }

    public FullyConnectedLayer(String path) throws FileManagingException{
        init(path);
    }

    /**
     * Empty constructor needed for Jackson
     */
    @SuppressWarnings("unused")
    public FullyConnectedLayer(){}

    @Override
    public float[][] process(float[][] input) {
        float[][] outputs = new float[input.length][];
        float[] layerInput;
        float[] layerOutput = null;

        for (int i = 0; i < input.length; i++) {
            layerInput = input[i];
            for (int j = 0; j < weights.length; j++) {
                layerOutput = new float[weights[j].length];
                for (int k = 0; k < weights[j].length; k++) {
                    layerOutput[k] = 0;
                    for (int l = 0; l < weights[j][k].length; l++) {
                        layerOutput[k] += layerInput[l] * weights[j][k][l];
                    }
                    layerOutput[k] += biases[j][k];
                    layerOutput[k] = activationFunction.activate(layerOutput[k]);
                }
                layerInput = layerOutput;
            }
            assert layerOutput != null;
            outputs[i] = Arrays.copyOf(layerOutput, layerOutput.length);
        }
        return outputs;
    }

    @Override
    public void init(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
            mapper.readerForUpdating(this).readValue(new File(path), FullyConnectedLayer.class);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred while trying to init the " +
                    "FullyConnectedLayer from file" + path + ": " + e.getLocalizedMessage());
        }
    }

    @Override
    public void init(RandomWeightGenerator randomGen) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    if (i == 0) {
                        weights[i][j][k] = randomGen.getWeight(inputLayerSize, weights[0].length);
                    } else if (i == weights.length - 1) {
                        weights[i][j][k] = randomGen.getWeight(weights[weights.length - 1].length, outputLayerSize);
                    } else {
                        weights[i][j][k] = randomGen.getWeight(weights[i].length, weights[i + 1].length);
                    }
                }
            }
        }
    }

    @Override
    public void save(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.writeValue(new File(path), this);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred trying to save the values of " +
                    "the Dataset" + e.getLocalizedMessage());
        }
    }

    @Override
    public void train(Dataset data, Model model) throws IncorrectDataException {
        float m_hat, v_hat, g;
            time++;
            for (int j = 0; j < weights.length; j++) {
                for (int k = 0; k < weights[j].length; k++) {
                    for (int l = 0; l < weights[j][k].length; l++) {
                        g = calculateWeightGradient(j, k, l, data);
                        m_weight[j][k][l] = beta_1 * m_weight[j][k][l] + beta_3 * g;
                        v_weight[j][k][l] = (float) (beta_2 * v_weight[j][k][l] + beta_4 * Math.pow(g, 2));
                        m_hat = (float) (m_weight[j][k][l] / (1 - Math.pow(beta_1, time)));
                        v_hat = (float) (v_weight[j][k][l] / (1 - Math.pow(beta_2, time)));
                        weights[j][k][l] = (float) (weights[j][k][l] - m_hat * (alpha / (Math.sqrt(v_hat) + epsilon)));
                    }
                }
            }
            for (int j = 0; j < biases.length; j++) {
                for (int k = 0; k < biases[j].length; k++) {
                    g = calculateBiasGradient(j, k, data);
                    m_bias[j][k] = beta_1 * m_bias[j][k] + beta_3 * g;
                    v_bias[j][k] = (float) (beta_2 * v_bias[j][k] + beta_4 * Math.pow(g, 2));
                    m_hat = (float) (m_bias[j][k] / (1 - Math.pow(beta_1, time)));
                    v_hat = (float) (v_bias[j][k] / (1 - Math.pow(beta_2, time)));
                    biases[j][k] = (float) (biases[j][k] - m_hat * (alpha / (Math.sqrt(v_hat) + epsilon)));
                }
            }
    }

    @Override
    public void initAdamValues(){
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
        loadHyperParams();
    }

    private void loadHyperParams(){
        alpha = trainingSettings.getLearningRate();
        beta_1 = trainingSettings.getExponentialDecayRateOne();
        beta_2 = trainingSettings.getExponentialDecayRateTwo();
        epsilon = trainingSettings.getEpsilon();
        beta_3 = 1 - beta_1;
        beta_4 = 1 - beta_2;
        time = 0;
    }

    private void allocateParams(){
        if (hiddenLayersSizes != null && hiddenLayersSizes.length != 0){
            weights = new float[1 + hiddenLayersSizes.length][][];
            weights[0] = new float[hiddenLayersSizes[0]][inputLayerSize];
            weights[weights.length - 1] = new float[outputLayerSize][hiddenLayersSizes[hiddenLayersSizes.length-1]];
            for (int i = 1; i < weights.length - 1; i++) {
                weights[i] = new float[hiddenLayersSizes[i]][hiddenLayersSizes[i-1]];
            }
            biases = new float[hiddenLayersSizes.length + 1][];
            for (int i = 0; i < biases.length - 1; i++) {
                biases[i] = new float[hiddenLayersSizes[i]];
            }
            biases[biases.length - 1] = new float[outputLayerSize];
        }else{
            weights = new float[1][outputLayerSize][inputLayerSize];
            biases = new float[1][outputLayerSize];
        }
    }

    private float calculateWeightGradient(int layer, int to, int from, Dataset data) {
        float original = weights[layer][to][from];
        float nudge = (float) Math.max(1e-6, Math.abs(original) * 1e-6);
        weights[layer][to][from] = original + nudge;
        float positiveNudge = fullModel.calculateAverageCost(data);
        weights[layer][to][from] = original - nudge;
        float negativeNudge = fullModel.calculateAverageCost(data);
        float gradient = (positiveNudge - negativeNudge) / ( 2 * nudge);
        weights[layer][to][from] = original;
        return gradient;
    }

    private float calculateBiasGradient(int layer, int neuron, Dataset data) {
        float original = biases[layer][neuron];
        float nudge = (float) Math.max(1e-6, Math.abs(original) * 1e-6);
        biases[layer][neuron] = original + nudge;
        float positiveNudge = fullModel.calculateAverageCost(data);
        biases[layer][neuron] = original - nudge;
        float negativeNudge = fullModel.calculateAverageCost(data);
        float gradient = (positiveNudge - negativeNudge) / (2 * nudge);
        biases[layer][neuron] = original;
        return gradient;
    }

    //region Getters and Setters
    @Override
    public TrainingSettings getTrainingSettings() {
        return trainingSettings;
    }

    @Override
    public void setTrainingSettings(TrainingSettings settings) {
        trainingSettings = settings;
        loadHyperParams();
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public Model getFullModel() {
        return fullModel;
    }

    @Override
    public void setFullModel(Model fullModel) {
        this.fullModel = fullModel;
    }
    //endregion

    //region Weight, Bias and Dimension Getters and Setters for Jackson

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

    public int getInputLayerSize() {
        return inputLayerSize;
    }

    public void setInputLayerSize(int inputLayerSize) {
        this.inputLayerSize = inputLayerSize;
    }

    public int getOutputLayerSize() {
        return outputLayerSize;
    }

    public void setOutputLayerSize(int outputLayerSize) {
        this.outputLayerSize = outputLayerSize;
    }

    public int[] getHiddenLayersSizes() {
        return hiddenLayersSizes;
    }

    public void setHiddenLayersSizes(int[] hiddenLayersSizes) {
        this.hiddenLayersSizes = hiddenLayersSizes;
    }

    //endregion
}
