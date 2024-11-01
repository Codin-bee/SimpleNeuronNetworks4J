package com.codingbee.snn4j.neural_network;

import com.codingbee.snn4j.algorithms.AlgorithmManager;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.exceptions.MethodCallingException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

@SuppressWarnings("unused")
public class OptimizedMLP {
    private String networkPath;
    private double[][][] weights;
    private double[][] biases;
    private final int[] hiddenLayersSizes;
    private final int inputLayerSize;
    private final int outputLayerSize;
    private boolean initialized = false;
    private TrainingSettings trainingSettings = new TrainingSettings();
    private DebuggingSettings debuggingSettings = new DebuggingSettings();

    /**
     *Creates new MLP(Multi-layer perceptron) based on the parameters it is given.
     * @param inputLayerSize the number of neurons in the input layer. Must be higher than 0.
     * @param outputLayerSize the number of neurons in the output layer. Must be higher than 0.
     * @param hiddenLayersSizes array of numbers which determine how many neurons will each hidden layer have. If it is null no hidden layer will be created. Every value of array must be higher than zero.
     * @param networkPath path, where the weights and biases will be stored in the file system. Needs to be unique or else the MLPs with the same name in file system will overwrite each other's values.Can not be an empty String("").
     *
     * @throws IncorrectDataException if requirements of any parameter are not fulfilled.
     */
    @SuppressWarnings("unused")
    public OptimizedMLP(int inputLayerSize, int outputLayerSize, int[] hiddenLayersSizes, String networkPath) throws IncorrectDataException {
        if (inputLayerSize<1) throw new IncorrectDataException("MLP constructor - input layer size must be higher than zero");
        if (outputLayerSize<1) throw new IncorrectDataException("MLP constructor - output layer size must be higher than zero");
        if (hiddenLayersSizes!=null) {
            for (int hiddenLayerSize : hiddenLayersSizes) {
                if (hiddenLayerSize < 1) throw new IncorrectDataException("MLP constructor - all hidden layers sizes must be higher than zero");
            }
        }
        if (networkPath.isEmpty()) throw new IncorrectDataException("OptimizedMLP constructor - network path must not be an empty String");

        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.hiddenLayersSizes = hiddenLayersSizes;
        this.networkPath = networkPath;
        initWeightMatrices();
    }

    //region Initialization and Saving

    /**
     * Initializes the networks values(weights and biases) from given directory
     * @param path path to the directory, where values are saved
     * @throws FileManagingException if any error occurs while working with files
     */
    public void initializeFromFiles(String path) throws FileManagingException {
        for (int i = 0; i < weights.length; i++) {
            try(BufferedReader reader = new BufferedReader(new FileReader(path + "/weights/w" + i + ".txt"))){
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        String[] values = reader.readLine().split(" ");
                        weights[i][j][k] = Double.parseDouble(values[k]);
                    }
                }
            }catch(Exception e){
                throw new FileManagingException(e.getLocalizedMessage());
            }
        }

        try(BufferedReader reader = new BufferedReader(new FileReader(path + "/biases.txt"))){
            for (int i = 0; i < biases.length; i++) {
                for (int j = 0; j < biases[i].length; j++) {
                    String[] values = reader.readLine().split(" ");
                    biases[i][j] = Double.parseDouble(values[j]);
                }
            }
        }catch(Exception e){
            throw new FileManagingException(e.getLocalizedMessage());
        }
        initialized = true;
    }

    /**
     * Initializes the networks values(weights and biases) from the model save path specified in the constructor
     * @throws FileManagingException if any error occurs while working with files
     */
    @SuppressWarnings("unused")
    public void initializeFromFiles() throws FileManagingException{
        initializeFromFiles(networkPath);
    }

    /**
     * Saves the networks values to the given path
     * @param path path to the directory where values will be saved
     * @throws FileManagingException if any exception occurs while working with files
     */
    @SuppressWarnings("unused")
    public void saveToFiles(String path) throws FileManagingException {
        for (int i = 0; i < weights.length; i++) {
            try(BufferedWriter writer = new BufferedWriter(new FileWriter(path + "/weights/w" + i + ".txt"))){
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        writer.write(weights[i][j][k] + " ");
                    }
                }
                writer.newLine();
            }catch (Exception e){
                throw new FileManagingException(e.getLocalizedMessage());
            }
        }
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(path + "/biases.txt"))) {
            for (double[] layerBiases : biases) {
                for (double bias : layerBiases) {
                    writer.write(bias + " ");
                }
                writer.newLine();
            }
        } catch (Exception e) {
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    /**
     * Saves the network values to the save path defined in the constructor
     * @throws FileManagingException if any error occurs while working with files
     */
    @SuppressWarnings("unused")
    public void saveToFiles() throws FileManagingException {
        saveToFiles(networkPath);
    }

    /**
     * Initializes the networks values(weights and biases) using random weight generator passed as a parameter
     * @param gen RandomWeightGenerator used to generate all random weights and biases
     */
    @SuppressWarnings("unused")
    public void initializeWithRandomValues(RandomWeightGenerator gen){
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = gen.getWeight();
                }
            }
        }
        for (int i = 0; i < biases.length - 1; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                biases[i][j] = gen.getHiddenLayerBias();
            }
        }
        for (int i = 0; i < biases[biases.length - 1].length; i++) {
            biases[biases.length - 1][i] = gen.getOutputLayerBias();
        }
        initialized = true;
    }
    //endregion

    //region Processing

    /**
     * Processes given values and returns activations of neurons in output layer
     * @param input the values to be processed
     * @return activations of output neurons
     * @throws MethodCallingException if the network has not been initialized yet
     */
    @SuppressWarnings("unused")
    public double[] processAsValues(double[] input) throws MethodCallingException {
        if (initialized) {
            double[] layerInput = input;
            double[] layerOutput = null;
            for (int i = 0; i < weights.length; i++) {
                layerOutput = new double[weights[i].length];
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        layerOutput[j] = layerInput[k] * weights[i][j][k];
                    }
                    layerOutput[j] += biases[i][j];
                    layerOutput[j] = leakyReLU(layerOutput[j], 0.001);
                }
                layerInput = layerOutput;
            }
            return layerOutput;
        }else{
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
    }

    /**
     * Processes given values and return the probabilities for each of output neurons
     * @param input the values to be processed
     * @return probabilities for each of the output neurons
     * @throws MethodCallingException if the network has not been initialized yet
     */
    @SuppressWarnings("unused")
    public double[] processAsProbabilities(double[] input) throws MethodCallingException {
        double[] probabilities = processAsValues(input);
        softmax(probabilities, 1);
        return probabilities;
    }

    /**
     * Processes given values and returns the index of output neuron with the highest activation
     * @param input the values to be processed
     * @return index of the output neuron with the highest activation
     * @throws MethodCallingException if the network has not been initialized yet
     */
    @SuppressWarnings("unused")
    public int processAsIndex(double[] input) throws MethodCallingException {
        return AlgorithmManager.getIndexWithHighestVal(processAsValues(input));
    }
    //endregion

    //region Training and analyzing
    @SuppressWarnings("unused")
    public void train(Dataset data, int epochs, boolean debugMode) throws MethodCallingException {
        double alpha = trainingSettings.getLearningRate();
        double beta_1 = trainingSettings.getExponentialDecayRateOne();
        double beta_2 = trainingSettings.getExponentialDecayRateTwo();
        double epsilon = trainingSettings.getEpsilon();
        double beta_3 = 1 - beta_1;
        double beta_4 = 1 - beta_2;
        double time = 0, m_hat, v_hat, g;
        double[][][] m_weight = new double[weights.length][][];
        double[][][] v_weight = new double[weights.length][][];
        for (int i = 0; i < m_weight.length; i++) {
            m_weight[i] = new double[weights[i].length][];
            v_weight[i] = new double[weights[i].length][];
            for (int j = 0; j < m_weight[i].length; j++) {
                m_weight[i][j] = new double[weights[i][j].length];
                v_weight[i][j] = new double[weights[i][j].length];
            }
        }
        double[][] m_bias = new double[biases.length][];
        double[][] v_bias = new double[biases.length][];
        for (int i = 0; i < m_bias[i].length; i++) {
            m_bias[i] = new double[biases[i].length];
            v_bias[i] = new double[biases[i].length];
        }
        for (int i = 0; i < epochs; i++) {
            time++;
            for (int j = 0; j < weights.length; j++) {
                for (int k = 0; k < weights[i].length; k++) {
                    for (int l = 0; l < weights[i][j].length; l++) {
                        g = calculateWeightGradient(j, k, l, data);
                        m_weight[j][k][l] = beta_1 * m_weight[j][k][l] + beta_3 * g;
                        v_weight[j][k][l] = beta_2 * v_weight[j][k][l] + beta_4 * Math.pow(g, 2);
                        m_hat = m_weight[j][k][l] / (1 - Math.pow(beta_1, time));
                        v_hat = v_weight[j][k][l] / (1 - Math.pow(beta_2, time));
                        m_weight[j][k][l] = m_weight[j][k][l] - m_hat * (alpha / (Math.sqrt(v_hat) + epsilon));
                    }
                }
            }
            for (int j = 0; j < biases.length; j++) {
                for (int k = 0; k < biases[i].length; k++) {
                    g = calculateBiasGradient(j, k, data);
                    m_bias[i][j] = beta_1 * m_bias[i][j] + beta_3 * g;
                    v_bias[i][j] = beta_2 * v_bias[i][j] + beta_4 * Math.pow(g, 2);
                    m_hat = m_bias[i][j] / (1 - Math.pow(beta_1, time));
                    v_hat = v_bias[i][j] / (1 - Math.pow(beta_2, time));
                    m_bias[i][j] = m_bias[i][j] - m_hat * (alpha / (Math.sqrt(v_hat) + epsilon));
                }
            }
        }
    }

    public double calculateAverageCost(Dataset data) throws MethodCallingException {
        double cost = 0;
        double count = 0;
        for (int i = 0; i < data.getInputData().length; i++) {
            double[] output = processAsProbabilities(data.getInputData()[i]);
            for (int j = 0; j < data.getInputData()[i].length; j++) {
                cost += output[j] * data.getExpectedResults()[i][j];
            }
            count++;
        }
        return cost / count;
    }

    @SuppressWarnings("unused")
    public double getCorrectPercentage(Dataset data) throws MethodCallingException {
        double percentage = 0;
        double count = 0;
        for (int i = 0; i < data.getInputData().length; i++) {
            int receivedIndex = processAsIndex(data.getInputData()[i]);
            int correctIndex = AlgorithmManager.getIndexWithHighestVal(data.getExpectedResults()[i]);
            count++;
            if (receivedIndex == correctIndex){
                percentage++;
            }
        }
        return (percentage / count) * 100;
    }
    //endregion

    //region Private Methods
    private void initWeightMatrices(){
        if (hiddenLayersSizes != null){
            weights = new double[1 + hiddenLayersSizes.length][][];
            weights[0] = new double[hiddenLayersSizes[0]][inputLayerSize];
            weights[hiddenLayersSizes.length] = new double[outputLayerSize][hiddenLayersSizes[hiddenLayersSizes.length-1]];
            for (int i = 1; i < weights.length - 1; i++) {
                weights[i] = new double[hiddenLayersSizes[i]][hiddenLayersSizes[i-1]];
            }
            biases = new double[hiddenLayersSizes.length + 1][];
            for (int i = 0; i < biases.length - 1; i++) {
                biases[i] = new double[hiddenLayersSizes[i]];
            }
            biases[hiddenLayersSizes.length] = new double[outputLayerSize];
        }else{
            weights = new double[1][inputLayerSize][outputLayerSize];
            biases = new double[0][outputLayerSize];
        }
    }

    @SuppressWarnings("all")
    private void softmax(double[] values, double temp){
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.exp(values[i] / temp);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }

    @SuppressWarnings("all")
    private double leakyReLU(double x, double alpha){
        if(x < 0){
            return 0;
        }
        return x * alpha;
    }

    private double calculateWeightGradient(int layer, int to, int from, Dataset data) throws MethodCallingException {
        double nudge = 0.0000001;
        double original = weights[layer][to][from];
        weights[layer][to][from] = original + nudge;
        double positiveNudge = calculateAverageCost(data);
        weights[layer][to][from] = original - nudge;
        double negativeNudge = calculateAverageCost(data);
        double gradient = (positiveNudge - negativeNudge) / 2 * nudge;
        weights[layer][to][from] = original;
        return gradient;
    }

    private double calculateBiasGradient(int layer, int neuron, Dataset data) throws MethodCallingException {
        double nudge = 0.0000001;
        double original = biases[layer][neuron];
        biases[layer][neuron] = original + nudge;
        double positiveNudge = calculateAverageCost(data);
        biases[layer][neuron] = original - nudge;
        double negativeNudge = calculateAverageCost(data);
        double gradient = (positiveNudge - negativeNudge) / 2 * nudge;
        biases[layer][neuron] = original;
        return gradient;
    }

    //endregion

    //region Getters and Setters
    @SuppressWarnings("unused")
    public TrainingSettings getTrainingSettings() {
        return trainingSettings;
    }

    @SuppressWarnings("unused")
    public void setTrainingSettings(TrainingSettings trainingSettings) {
        this.trainingSettings = trainingSettings;
    }

    @SuppressWarnings("unused")
    public DebuggingSettings getDebuggingSettings() {
        return debuggingSettings;
    }

    @SuppressWarnings("unused")
    public void setDebuggingSettings(DebuggingSettings debuggingSettings) {
        this.debuggingSettings = debuggingSettings;
    }

    @SuppressWarnings("unused")
    public String getNetworkPath() {
        return networkPath;
    }

    @SuppressWarnings("unused")
    public void setNetworkPath(String networkPath) {
        this.networkPath = networkPath;
    }

    //endregion
}