package com.codingbee.snn4j.neural_network;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
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

    /**Creates new MLP(Multi-layer perceptron) based on the parameters it is given.
     *
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
        initWeightMatricies();
    }

    @SuppressWarnings("unused")
    public void initializeFromFiles() throws FileManagingException {
        for (int i = 0; i < weights.length; i++) {
            try(BufferedReader reader = new BufferedReader(new FileReader(networkPath + "/weights/w" + i + ".txt"))){
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

        try(BufferedReader reader = new BufferedReader(new FileReader(networkPath + "/biases.txt"))){
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

    @SuppressWarnings("unused")
    public void saveToFiles() throws FileManagingException {
        for (int i = 0; i < weights.length; i++) {
            try(BufferedWriter writer = new BufferedWriter(new FileWriter(networkPath + "/weights/w" + i + ".txt"))){
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
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(networkPath + "/biases.txt"))) {
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

    @SuppressWarnings("unused")
    public double[] processAsProbabilities(double[] input){
        double[] layerInput = input;
        double[] layerOutput = null;
        for (int i = 0; i < weights.length; i++) {
            layerOutput = new double[weights[i].length];
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    layerOutput[j] = layerInput[k] * weights[i][j][k];
                }
                layerOutput[j] += biases[i][j];
                //activate
            }
            layerInput = layerOutput;
        }
        return layerOutput;
    }
    //region Private Methods
    public void initWeightMatricies(){
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
