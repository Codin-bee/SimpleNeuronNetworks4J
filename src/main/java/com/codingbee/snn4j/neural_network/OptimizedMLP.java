package com.codingbee.snn4j.neural_network;

import com.codingbee.snn4j.algorithms.AlgorithmManager;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.exceptions.MethodCallingException;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

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


    /*Initialization and saving*/

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
            for (int i = 0; i < biases.length; i++) {
                for (int j = 0; j < biases[i].length; j++) {
                    writer.write(biases[i][j] + " ");
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


    /*Processing*/

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


    //cost, train, percentage
    //region Private Methods
    public void initWeightMatrices(){
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

    public void softmax(double[] values, double temp){
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.exp(values[i] / temp);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }

    public double leakyReLU(double x, double alpha){
        if(x < 0){
            return 0;
        }
        return x * alpha;
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
