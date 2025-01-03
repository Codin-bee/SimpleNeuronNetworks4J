package com.codingbee.snn4j.neural_networks.mlp;

import com.codingbee.snn4j.interface_implementations.activation_functions.ReLU;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.ActivationFunction;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;
import com.codingbee.snn4j.neural_networks.DebuggingSettings;
import com.codingbee.snn4j.neural_networks.TrainingSettings;
import com.codingbee.tool_box.algorithms.Algorithms;
import com.codingbee.tool_box.exceptions.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


@SuppressWarnings("unused")
public class OptimizedMLP {
    private String networkPath;
    private float[][][] weights;
    private float[][] biases;
    private final int[] hiddenLayersSizes;
    private final int inputLayerSize;
    private final int outputLayerSize;
    private boolean initialized = false;
    private TrainingSettings trainingSettings = new TrainingSettings();
    private DebuggingSettings debuggingSettings = new DebuggingSettings();
    private ActivationFunction activationFunction = new ReLU();


    /**
     *Creates new MLP(Multi-layer perceptron) based on the parameters it is given.
     * @param inputLayerSize the number of neurons in the input layer. Must be higher than 0.
     * @param outputLayerSize the number of neurons in the output layer. Must be higher than 0.
     * @param hiddenLayersSizes array of numbers which determine how many neurons will each hidden layer have. If it is null no hidden layer will be created. Every value of array must be higher than zero.
     * @param networkPath path, where the weights and biases will be stored in the file system. Needs to be unique or else the MLPs with the same name in file system will overwrite each other's values.Can not be an empty String("").
     *
     * @throws IncorrectDataException if requirements of any parameter are not fulfilled.
     */
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
        allocateWeightMatrices();
    }

    //region Initialization and Saving

    /**
     * Initializes the networks values(weights and biases) from given directory
     * @param path path to the directory, where values are saved
     * @throws FileManagingException if any error occurs while working with files
     */
    public void initializeFromFiles(String path) throws FileManagingException {
        for (int i = 0; i < weights.length; i++) {
            String filePath = path + "\\weights\\w" + i + ".txt";
            try(BufferedReader reader = new BufferedReader(new FileReader(filePath))){
                for (int j = 0; j < weights[i].length; j++) {
                    String[] values = reader.readLine().split(" ");
                    for (int k = 0; k < weights[i][j].length; k++) {
                        weights[i][j][k] = Float.parseFloat(values[k]);
                    }
                }
            }catch(Exception e){
                throw new FileManagingException(e.getLocalizedMessage() + "; at path: " + filePath);
            }
        }

        try(BufferedReader reader = new BufferedReader(new FileReader(path + "\\biases.txt"))){
            for (int i = 0; i < biases.length; i++) {
                String[] values = reader.readLine().split(" ");
                for (int j = 0; j < biases[i].length; j++) {
                    biases[i][j] = Float.parseFloat(values[j]);
                }
            }
        }catch(Exception e){
            throw new FileManagingException(e.getLocalizedMessage() + "; at path: " + path + "\\biases.txt");
        }
        initialized = true;
    }

    /**
     * Initializes the networks values(weights and biases) from the model save path specified in the constructor
     * @throws FileManagingException if any error occurs while working with files
     */
    public void initializeFromFiles() throws FileManagingException{
        initializeFromFiles(networkPath);
    }

    /**
     * Saves the networks values to the given path
     * @param path path to the directory where values will be saved
     * @throws FileManagingException if any exception occurs while working with files
     */
    public void saveToFiles(String path) throws FileManagingException, MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not be saved, because it has not been initialized yet");
        }
        try {
            //Ignored because false can be returned because of existing directories
            boolean mainCreated = new File(path).mkdirs();
            boolean weightCreated = new File(path + "\\weights").mkdirs();
        } catch (Exception e) {
            throw new FileManagingException("Could not create directories" + e.getLocalizedMessage());
        }

        for (int i = 0; i < weights.length; i++) {
            try(BufferedWriter writer = new BufferedWriter(new FileWriter(path + "\\weights\\w" + i + ".txt"))){
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        writer.write(weights[i][j][k] + " ");
                    }
                    writer.newLine();
                }
            }catch (Exception e){
                throw new FileManagingException(e.getLocalizedMessage());
            }
        }
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(path + "\\biases.txt"))) {
            for (float[] layerBiases : biases) {
                for (float bias : layerBiases) {
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
    public void saveToFiles() throws FileManagingException, MethodCallingException {
        saveToFiles(networkPath);
    }

    /**
     * Initializes the networks values(weights and biases) using random weight generator passed as a parameter
     * @param gen RandomWeightGenerator used to generate all random weights and biases
     */
    public void initializeWithRandomValues(RandomWeightGenerator gen) throws DevelopmentException {
        try {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        if (i == 0) {
                            weights[i][j][k] = (float) gen.getWeight(inputLayerSize, weights[0].length);
                        } else if (i == weights.length - 1) {
                            weights[i][j][k] = (float) gen.getWeight(weights[weights.length - 1].length, outputLayerSize);
                        } else {
                            weights[i][j][k] = (float) gen.getWeight(weights[i].length, weights[i + 1].length);
                        }
                    }
                }
            }
            for (int i = 0; i < biases.length - 1; i++) {
                for (int j = 0; j < biases[i].length; j++) {
                    biases[i][j] = (float) gen.getHiddenLayerBias();
                }
            }
            for (int i = 0; i < biases[biases.length - 1].length; i++) {
                biases[biases.length - 1][i] = (float) gen.getOutputLayerBias();
            }
            initialized = true;
        }catch (Exception e){
            throw new DevelopmentException("An error occurred because of wrong inside logic of method," +
                    " detailed description: " + e.getLocalizedMessage());
        }
    }
    //endregion

    //region Processing

    /**
     * Processes given values and returns activations of neurons in output layer
     * @param input the values to be processed
     * @return activations of output neurons
     * @throws MethodCallingException if the network has not been initialized yet
     */
    public float[] processAsValues(double[] input) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        float[] layerInput = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            layerInput[i] = (float) input[i];
        }
        float[] layerOutput = null;
        for (int i = 0; i < weights.length; i++) {
            layerOutput = new float[weights[i].length];
            for (int j = 0; j < weights[i].length; j++) {
                layerOutput[j] = 0;
                for (int k = 0; k < weights[i][j].length; k++) {
                    layerOutput[j] += layerInput[k] * weights[i][j][k];
                }
                layerOutput[j] += biases[i][j];
                layerOutput[j] = activationFunction.activate(layerOutput[j]);
            }
            layerInput = layerOutput;
        }
        return layerOutput;
    }

    /**
     * Processes given values and return the probabilities for each of output neurons
     * @param input the values to be processed
     * @return probabilities for each of the output neurons
     * @throws MethodCallingException if the network has not been initialized yet
     */
    public float[] processAsProbabilities(double[] input) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        float[] probabilities = processAsValues(input);
        Algorithms.softmaxInPlace(probabilities, 1);
        return probabilities;
    }

    /**
     * Processes given values and returns the index of output neuron with the highest activation
     * @param input the values to be processed
     * @return index of the output neuron with the highest activation
     * @throws MethodCallingException if the network has not been initialized yet
     */
    public int processAsIndex(double[] input) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        return Algorithms.getIndexWithHighestVal(processAsValues(input));
    }
    //endregion

    //region Training and analyzing

    /**
     * Trains the model using backpropagation for given amount of epochs
     * @param data the data for the model to be trained on
     * @param epochs number of epochs(iterations) of the training
     * @param debugMode whether to print debug info or not,the level of debugging is based on debugging
     *                  settings, see {@link DebuggingSettings} for details, and use {@link #setDebuggingSettings}
     * @throws MethodCallingException if the network has not been initialized or any of passed arguments are invalid
     */
    public void train(Dataset data, int epochs, boolean debugMode) throws MethodCallingException, FileManagingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        float alpha = (float) trainingSettings.getLearningRate();
        float beta_1 = (float) trainingSettings.getExponentialDecayRateOne();
        float beta_2 = (float) trainingSettings.getExponentialDecayRateTwo();
        float epsilon = (float) trainingSettings.getEpsilon();
        float beta_3 = 1 - beta_1;
        float beta_4 = 1 - beta_2;
        float time = 0, m_hat, v_hat, g;
        float[][][] m_weight = new float[weights.length][][];
        float[][][] v_weight = new float[weights.length][][];
        for (int i = 0; i < m_weight.length; i++) {
            m_weight[i] = new float[weights[i].length][];
            v_weight[i] = new float[weights[i].length][];
            for (int j = 0; j < m_weight[i].length; j++) {
                m_weight[i][j] = new float[weights[i][j].length];
                v_weight[i][j] = new float[weights[i][j].length];
            }
        }
        float[][] m_bias = new float[biases.length][];
        float[][] v_bias = new float[biases.length][];
        for (int i = 0; i < m_bias.length; i++) {
            m_bias[i] = new float[biases[i].length];
            v_bias[i] = new float[biases[i].length];
        }
        for (int i = 0; i < epochs; i++) {
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

            if (debuggingSettings.isEveryIterationPrint()){
                System.out.println("Iteration " + i + ":");
                System.out.println("Cost: " + calculateAverageCostInParallel(data, trainingSettings.getBatchSize()));
                System.out.println("Correct percentage: " + getCorrectPercentage(data));
            }
            if (debuggingSettings.isSavePeriodically()){
                saveToFiles();
            }
        }
        saveToFiles();
    }

    /**
     * Calculates the average cost of the network across all the given data.
     * @param data the dataset to calculate the cost on
     * @return the average cost of the network across all the data
     * @throws MethodCallingException when this method is called on network that hasn't been initialized yet
     */
    public float calculateAverageCost(Dataset data) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        return calculateAverageCost(data.getInputData(), data.getExpectedResults());
    }

    /**
     * Calculates the average cost of the network across all the given data.
     * @param inputData the input values for the calculations
     * @param expectedOutputData expected values of the processing
     * @return the average cost of the network across all the data
     * @throws MethodCallingException when this method is called on network that hasn't been initialized yet
     */
    public float calculateAverageCost(double[][] inputData, double[][] expectedOutputData) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        double cost = 0;
        for (int i = 0; i < inputData.length; i++) {
            cost += calculateCost(inputData[i], expectedOutputData[i]);
        }
        return (float) (cost / inputData.length);
    }

    /**
     * Calculates the average cost of the network across all the given data in parallel batches
     * @param inputData the input values for the calculations
     * @param expectedOutputData expected values of the processing
     * @param batchSize size of one batch of data
     * @return the average cost of the network across all the data
     * @throws RuntimeException if issue arises because of threads
     * @throws MethodCallingException when called without initializing network
     */
    public float calculateAverageCostInParallel(double[][] inputData, double[][] expectedOutputData, int batchSize) throws RuntimeException, MethodCallingException {
        if (!initialized) {
            throw new MethodCallingException("The network cannot process anything, because it has not been initialized yet");
        }

        int batchCount = (int) Math.ceil((double) inputData.length / batchSize);
        float cost = 0;

        ExecutorService executorService = Executors.newFixedThreadPool(Math.min(batchCount, Runtime.getRuntime().availableProcessors()));
        List<Future<Float>> futures = new ArrayList<>();

        try {
            for (int i = 0; i < batchCount; i++) {
                final int startI = i * batchSize;
                final int endI = Math.min(startI + batchSize, inputData.length);

                futures.add(executorService.submit(() -> calculateAverageCost(
                        Arrays.copyOfRange(inputData, startI, endI),
                        Arrays.copyOfRange(expectedOutputData, startI, endI))));
            }

            for (Future<Float> future : futures) {
                cost += future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("An error occurred during parallel computation", e);
        } finally {
            executorService.shutdown();
        }

        return cost / batchCount;
    }

    /**
     * Calculates cost of the network on given example.
     * @param data the dataset to calculate the cost on
     * @param batchSize size of one batch of data
     * @return the average cost of the network across all the data
     * @throws MethodCallingException when called without initializing network
     */
    public float calculateAverageCostInParallel(Dataset data, int batchSize) throws MethodCallingException {
        return calculateAverageCostInParallel(data.getInputData(), data.getExpectedResults(), batchSize);
    }

    /**
     * Calculates cost of the network on given example.
     * @param input input for the network
     * @param expectedOutput expected output of the network
     * @return value of cost function ran on given data
     * @throws MethodCallingException when the method is called on network that hasn't been initialized yet
     */
    public float calculateCost(double[] input, double[] expectedOutput) throws MethodCallingException {
        double cost = 0;
        float[] output = processAsValues(input);
        for (int i = 0; i < output.length; i++) {
            double diff = output[i] - expectedOutput[i];
            cost += Math.pow(diff, 2);
        }

        return (float) cost;
    }

    /**
     * Calculates the correctness percentage of the network on given data.
     * @param data the dataset to calculate the percentage on
     * @return the correctness percentage of the network on given dataset
     * @throws MethodCallingException when the method is called on network that hasn't been initialized yet
     */
    public float getCorrectPercentage(Dataset data) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        return getCorrectPercentage(data.getInputData(), data.getExpectedResults());
    }

    /**
     * Calculates the correctness percentage of the network on given data.
     * @param inputData the inputs for the network
     * @param expectedOutputData expected (correct) outputs of the network
     * @return the correctness percentage of the network on given data
     * @throws MethodCallingException when the method is called on network that hasn't been initialized yet
     */
    public float getCorrectPercentage(double[][] inputData, double[][] expectedOutputData) throws MethodCallingException {
        if (!initialized){
            throw new MethodCallingException("The network can not process anything, because it has not been initialized yet");
        }
        double percentage = 0;
        double count = 0;
        for (int i = 0; i < inputData.length; i++) {
            if (processAsIndex(inputData[i]) == Algorithms.getIndexWithHighestVal(expectedOutputData[i])){
                percentage++;
            }
            count++;
        }
        return (float) ((percentage / count) * 100);
    }
    //endregion

    //region Private Methods

    /**
     * Allocates the space needed for all weight matrices of the network.
     */
    private void allocateWeightMatrices(){
        if (hiddenLayersSizes != null){
            weights = new float[1 + hiddenLayersSizes.length][][];
            weights[0] = new float[hiddenLayersSizes[0]][inputLayerSize];
            weights[hiddenLayersSizes.length] = new float[outputLayerSize][hiddenLayersSizes[hiddenLayersSizes.length-1]];
            for (int i = 1; i < weights.length - 1; i++) {
                weights[i] = new float[hiddenLayersSizes[i]][hiddenLayersSizes[i-1]];
            }
            biases = new float[hiddenLayersSizes.length + 1][];
            for (int i = 0; i < biases.length - 1; i++) {
                biases[i] = new float[hiddenLayersSizes[i]];
            }
            biases[hiddenLayersSizes.length] = new float[outputLayerSize];
        }else{
            weights = new float[1][inputLayerSize][outputLayerSize];
            biases = new float[0][outputLayerSize];
        }
    }

    /**
     * Calculates gradient of the cost function with respect to the given weight, on the given data.
     * @param layer layer of the MLP where weight is located
     * @param to neuron index in next layer
     * @param from neuron index in the previous layer
     * @param data the data for calculating cost function
     * @return the gradient of the cost function with respect to the weight
     * @throws MethodCallingException when the method is called on network that hasn't been initialized yet
     */
    private float calculateWeightGradient(int layer, int to, int from, Dataset data) throws MethodCallingException {
        float original = weights[layer][to][from];
        float nudge = (float) Math.max(1e-6, Math.abs(original) * 1e-6);
        weights[layer][to][from] = original + nudge;
        float positiveNudge = calculateAverageCostInParallel(data, trainingSettings.getBatchSize());
        weights[layer][to][from] = original - nudge;
        float negativeNudge = calculateAverageCostInParallel(data, trainingSettings.getBatchSize());
        float gradient = (positiveNudge - negativeNudge) / ( 2 * nudge);
        weights[layer][to][from] = original;
        return gradient;
    }

    /**
     * Calculates gradient of the cost function with respect to the given bias, on the given data.
     * @param layer layer of the MLP where bias is located
     * @param neuron the index of the neuron the bias corresponds to in given layer
     * @param data the data for calculating cost function
     * @return the gradient of the cost function with respect to the bias
     * @throws MethodCallingException when the method is called on network that hasn't been initialized yet
     */
    private float calculateBiasGradient(int layer, int neuron, Dataset data) throws MethodCallingException {
        float original = biases[layer][neuron];
        float nudge = (float) Math.max(1e-6, Math.abs(original) * 1e-6);
        biases[layer][neuron] = original + nudge;
        float positiveNudge = calculateAverageCostInParallel(data, trainingSettings.getBatchSize());
        biases[layer][neuron] = original - nudge;
        float negativeNudge = calculateAverageCostInParallel(data, trainingSettings.getBatchSize());
        float gradient = (positiveNudge - negativeNudge) / (2 * nudge);
        biases[layer][neuron] = original;
        return gradient;
    }

    //endregion

    //region Getters and Setters

    /**
     * Returns the training setting of the network.
     * @return the training setting of the network
     */
    public TrainingSettings getTrainingSettings() {
        return trainingSettings;
    }

    /**
     * Sets the training setting of the network.
     * @param trainingSettings the new training settings
     */
    public void setTrainingSettings(TrainingSettings trainingSettings) {
        this.trainingSettings = trainingSettings;
    }

    /**
     * Returns the debugging settings of the network.
     * @return the debugging settings of the network.
     */
    public DebuggingSettings getDebuggingSettings() {
        return debuggingSettings;
    }

    /**
     * Sets new debugging settings of the network.
     * @param debuggingSettings the new debugging settings
     */
    public void setDebuggingSettings(DebuggingSettings debuggingSettings) {
        this.debuggingSettings = debuggingSettings;
    }

    /**
     * Returns the path the network uses to save its files such as weight matrices.
     * @return the path to network files
     */
    public String getNetworkPath() {
        return networkPath;
    }

    /**
     * Sets new network path to store the network files such as weight matrices.
     * @param networkPath the new network file path
     */
    public void setNetworkPath(String networkPath) {
        this.networkPath = networkPath;
    }

    /**
     * Returns the activation function the network uses between each neuron layer
     * @return the activation function used by the network
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     * Sets new activation function for the network to use between each neuron layer
     * @param activationFunction the new activation function
     */
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
    //endregion
}