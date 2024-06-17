package com.codingbee.snn4j.neural_network;

import com.codingbee.snn4j.enums.TrainingDataFormat;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.objects_for_parsing.TrainingExample;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Network {
    private final int networkNo;
    private final List<List<Neuron>> hiddenLayers;
    private final List<Neuron> outputLayer;
    private final int[] hiddenLayersSizes;
    private final int inputLayerSize;
    private final int outputLayerSize;

    /**Creates new Network object based on the parameters it is given.
     *
     * @param inputLayerSize amount of parameters passed to the input layer of the network. Must be higher than 0.
     * @param outputLayerSize the number of outcomes in the output layer. Must be higher than 0.
     * @param hiddenLayersSizes array of numbers which determine how many neurons will each hidden layer have. If it is null no hidden layer will be crated.
     * @param networkNo number which the network will be assigned to in the file system. Needs to be unique from the rest to work properly.
     *
     * @throws IncorrectDataException if requirements of the parameters are not fulfilled.
     */
    public Network(int inputLayerSize, int outputLayerSize, int[] hiddenLayersSizes, int networkNo) throws IncorrectDataException {
        if (inputLayerSize<1) throw new IncorrectDataException("Network constructor - input layer size");
        if (outputLayerSize<1) throw new IncorrectDataException("Network constructor - output layer size");
        if (hiddenLayersSizes!=null) for (int hiddenLayerSize : hiddenLayersSizes) {
            if (hiddenLayerSize < 1) throw new IncorrectDataException("Network constructor - hidden layers sizes");
        }
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.hiddenLayersSizes = hiddenLayersSizes;
        this.networkNo = networkNo;
        hiddenLayers = new ArrayList<>();
        outputLayer = new ArrayList<>();

    }

    /**
     * Generates random values for neuron initialization.
     * @param dirPath path where directories will be generated
     * @param initAfterwards boolean value, if true neurons will be initialized after values were created
     * @throws FileManagingException if some problem arises while working with files
     */
    public void createRandomNeuronValuesInDir(String dirPath, boolean initAfterwards) throws FileManagingException {
        try {
            Random rand = new Random();
            Files.createDirectories(Paths.get(dirPath + "/neural_networks/network" + networkNo + "/layers/layer0"));
            for (int i = 0; i < hiddenLayersSizes[0]; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + "/neural_networks/network" + networkNo + "/layers/layer0/neuron" + i + ".txt"));
                writer.write(String.valueOf(Math.random()));
                for (int j = 0; j < inputLayerSize; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(rand.nextGaussian() * Math.sqrt(2.0 / inputLayerSize)));
                }
                writer.close();
            }
            for (int i = 1; i < hiddenLayersSizes.length; i++) {
                Files.createDirectories(Paths.get(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + i));

                for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                    BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + i + "/neuron" + j + ".txt"));
                    writer.write(String.valueOf(Math.random()));
                    for (int k = 0; k < hiddenLayersSizes[i-1]; k++) {
                        writer.newLine();
                        writer.write(String.valueOf(rand.nextGaussian() * Math.sqrt(2.0 / inputLayerSize)));
                    }
                    writer.close();
                }
            }
            Files.createDirectories(Paths.get(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + hiddenLayersSizes.length));
            for (int i = 0; i < outputLayerSize; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + "/neural_networks/network" + networkNo + "/layers/layer"
                        + hiddenLayersSizes.length + "/neuron" + i + ".txt"));
                writer.write(String.valueOf(Neuron.LAST));
                for (int j = 0; j < hiddenLayersSizes[hiddenLayersSizes.length-1]; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(rand.nextGaussian() * Math.sqrt(2.0 / inputLayerSize)));
                }
                writer.close();
            }
            if (initAfterwards){
                initNeuronsFromDir(dirPath);
            }
        } catch (IOException e) {
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    /**
     * Initializes the network's neurons.
     * @param dirPath path where directories with files are located
     * @throws FileManagingException if some problem arises while working with files
     */
    public void initNeuronsFromDir(String dirPath) throws FileManagingException {
        try {
            //FIRST HIDDEN

            List<Neuron> tempNeurons = new ArrayList<>();
            for (int j = 0; j < hiddenLayersSizes[0]; j++) {
                BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + 0
                        + "/neuron" + j + ".txt"));
                double bias = Double.parseDouble(reader.readLine());
                double[] weights = new double[inputLayerSize];

                for (int k = 0; k < inputLayerSize; k++) {
                    weights[k] = Double.parseDouble(reader.readLine());
                }
                tempNeurons.add(new Neuron(weights, bias));
            }
            hiddenLayers.add(tempNeurons);


            //HIDDEN
            for (int i = 1; i < hiddenLayersSizes.length; i++) {
                List<Neuron> tempNeurons2 = new ArrayList<>();
                for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                    BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + i
                        + "/neuron" + j + ".txt"));
                    double bias = Double.parseDouble(reader.readLine());
                    double[] weights = new double[hiddenLayersSizes[i-1]];

                    for (int k = 0; k < hiddenLayersSizes[i-1]; k++) {
                        weights[k] = Double.parseDouble(reader.readLine());
                    }
                    tempNeurons2.add(new Neuron(weights, bias));
                }
                hiddenLayers.add(tempNeurons2);
            }

            //OUTPUT
            for (int i = 0; i < outputLayerSize; i++) {
                BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + hiddenLayersSizes.length + "/neuron" + i + ".txt"));
                double bias = Double.parseDouble(reader.readLine());
                double[] weights = new double[hiddenLayersSizes[hiddenLayersSizes.length-1]];
                for (int k = 0; k < weights.length; k++) {
                    weights[k] = Double.parseDouble(reader.readLine());
                }
                outputLayer.add(new Neuron(weights, bias));
            }


        }catch (IOException e){
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    /**
     * Processes given values through the network and returns the networks decision about similarity with training data.
     * @param values array of doubles which represent some variables depending on your application
     * @return array same length as the output layer defined in constructor, each value will be between 0 and 1 depending on its
     * probability to be correct, which means higher value, higher probability
     */
    public double[] process(double[] values){
        double[] values2;
        for (int i = 0; i < hiddenLayersSizes.length; i++) {
            values2 = new double[hiddenLayersSizes[i]];
            for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                hiddenLayers.get(i).get(j).processNums(values);
                values2[j] = hiddenLayers.get(i).get(j).getFinalValue();
            }
            values = values2;
        }
        values2 = new double[outputLayerSize];
        for (int i = 0; i < outputLayerSize; i++) {
            outputLayer.get(i).processNums(values);
            values2[i] = sigmoid(outputLayer.get(i).getFinalValue());
        }
        return values2;
    }

    /**
     * Trains the network to act more like you want based on the training examples. It corrects the values, but doesn't
     * overwrite the values in files so if you want to keep them you should call {@link #saveNetworksValues saveNetworkValues()}
     * @param data data, which is the network trained on
     * @param learningRate double deciding how big changes should be done to the weights
     * @param printCosts whether to print starting and final cost function or not
     */
    public void trainWeights(Dataset data, double learningRate, boolean printCosts){
        double[][] trainingDataSet = data.getInputData();
        double[][] expectedResults = data.getExpectedResults();

        if (printCosts) {
            System.out.println("Starting cost: " + calculateAverageCost(trainingDataSet, expectedResults));
        }
        for (int i = 0; i < hiddenLayersSizes.length; i++) {
            for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                for (int k = 0; k < hiddenLayers.get(i).get(j).getWeights().length; k++) {
                    double originalCost = calculateAverageCost(trainingDataSet, expectedResults);
                    double originalWeight = hiddenLayers.get(i).get(j).getWeight(k);
                    hiddenLayers.get(i).get(j).setWeight(k, originalWeight + learningRate);
                    double costWithFirstNudge = calculateAverageCost(trainingDataSet, expectedResults);
                    if (originalCost > costWithFirstNudge) {
                        break;
                    } else {
                        hiddenLayers.get(i).get(j).setWeight(k, originalWeight - learningRate);
                        double costWithRevertedNudge = calculateAverageCost(trainingDataSet, expectedResults);
                        if (costWithRevertedNudge < originalCost) {
                            break;
                        } else {
                            hiddenLayers.get(i).get(j).setWeight(k, originalWeight);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < outputLayerSize; i++) {
            for (int j = 0; j < outputLayer.get(i).getWeights().length; j++) {
                double originalCost = calculateAverageCost(trainingDataSet, expectedResults);
                double originalWeight = outputLayer.get(i).getWeight(j);
                outputLayer.get(i).setWeight(j, originalWeight + learningRate);
                double costWithFirstNudge = calculateAverageCost(trainingDataSet, expectedResults);
                if (originalCost > costWithFirstNudge) {
                    break;
                } else {
                    outputLayer.get(i).setWeight(j, originalWeight - learningRate);
                    double costWithRevertedNudge = calculateAverageCost(trainingDataSet, expectedResults);
                    if (costWithRevertedNudge < originalCost) {
                        break;
                    } else {
                        outputLayer.get(i).setWeight(j, originalWeight);
                    }
                }
            }
        }
        if (printCosts) {
            System.out.println("Cost after training: " + calculateAverageCost(trainingDataSet, expectedResults));
    }
    }

    public void trainBiases(Dataset data, double learningRate, boolean printCosts){
        double[][] trainingDataSet = data.getInputData();
        double[][] expectedResults = data.getExpectedResults();
        if (printCosts) {
            System.out.println("Starting cost: " + calculateAverageCost(trainingDataSet, expectedResults));
        }
        for (int i = 0; i < hiddenLayersSizes.length; i++) {
            for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                    double originalCost = calculateAverageCost(trainingDataSet, expectedResults);
                    double originalBias = hiddenLayers.get(i).get(j).getBias();
                    hiddenLayers.get(i).get(j).setBias(originalBias + learningRate);
                    double costWithFirstNudge = calculateAverageCost(trainingDataSet, expectedResults);
                    if (originalCost > costWithFirstNudge) {
                        break;
                    } else {
                        hiddenLayers.get(i).get(j).setBias(originalBias - learningRate);
                        double costWithRevertedNudge = calculateAverageCost(trainingDataSet, expectedResults);
                        if (costWithRevertedNudge < originalCost) {
                            break;
                        } else {
                            hiddenLayers.get(i).get(j).setBias(originalBias);
                        }
                    }
            }
        }
        for (int i = 0; i < outputLayerSize; i++) {
            double originalCost = calculateAverageCost(trainingDataSet, expectedResults);
            double originalBias = outputLayer.get(i).getBias();
            outputLayer.get(i).setBias(originalBias + learningRate);
            double costWithFirstNudge = calculateAverageCost(trainingDataSet, expectedResults);
            if (originalCost > costWithFirstNudge) {
                break;
            } else {
                outputLayer.get(i).setBias(originalBias - learningRate);
                double costWithRevertedNudge = calculateAverageCost(trainingDataSet, expectedResults);
                if (costWithRevertedNudge < originalCost) {
                    break;
                } else {
                    outputLayer.get(i).setBias(originalBias);
                }
            }
        }
        if (printCosts) {
            System.out.println("Cost after training: " + calculateAverageCost(trainingDataSet, expectedResults));
        }

    }

    /**
     * Calculates cost function of given data.
     * @param trainingDataSet 2D array of training values
     * @param expectedResults 2D array of results you expect to get after processing the training data set
     * @return the average cost function
     */
    public double calculateAverageCost(double[][] trainingDataSet, double[][] expectedResults){
        double costsSummed = 0;
        int numberOfCostsInSum = 0;
        for (int i =0; i < trainingDataSet.length; i++) {
            double[] received = process(trainingDataSet[i]);
            for (int j = 0; j < outputLayerSize; j++) {
                costsSummed += Math.pow(received[j] - expectedResults[i][j], 2);
            }
            numberOfCostsInSum++;
        }
        return costsSummed/numberOfCostsInSum;
    }

    /**
     * Loads training data into given object.
     * @param directoryPath path to directory with training data
     * @param dataFormat enum deciding how to read the files. More information in {@link TrainingDataFormat}
     * @param data object holding arrays with the data, this method writes values directly into the object
     * @throws FileManagingException if some problem arises while working with files
     */
    public void loadData(String directoryPath, TrainingDataFormat dataFormat, Dataset data) throws FileManagingException {
        double[][] trainingDataSet;
        double[][] expectedResults;
        try {
            switch (dataFormat) {
                case JSON_ONE -> {
                    ObjectMapper mapper = new ObjectMapper();
                    File directory = new File(directoryPath);
                    File[] files = directory.listFiles((dir, name) -> name.startsWith("example") && name.endsWith("json"));
                    if (files != null) {
                        trainingDataSet = new double[files.length][inputLayerSize];
                        expectedResults = new double[files.length][outputLayerSize];
                        for (int i = 0; i < files.length; i++) {
                            Arrays.fill(expectedResults[i], 0);
                            TrainingExample example = mapper.readValue(new File(directoryPath + "/example" + i + ".json"), TrainingExample.class);
                            trainingDataSet[i] = example.getValues();
                            expectedResults[i][example.getCorrectNeuronIndex()] = 1;
                        }

                        data.setInputData(trainingDataSet);
                        data.setExpectedResults(expectedResults);
                    }
                }
            }
        }catch (IOException e){
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    /**
     * Saves networks values back to files.
     * @param directoryPath path to the files
     * @throws FileManagingException if some problem arises while working with files
     */
    public void saveNetworksValues(String directoryPath) throws FileManagingException{
        try {
            Files.createDirectories(Paths.get(directoryPath + "/neural_networks/network" + networkNo + "/layers/layer0"));
            for (int i = 0; i < hiddenLayersSizes[0]; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(directoryPath + "/neural_networks/network" + networkNo + "/layers/layer0/neuron" + i + ".txt"));
                writer.write(String.valueOf(hiddenLayers.getFirst().get(i).getBias()));
                for (int j = 0; j < inputLayerSize; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(hiddenLayers.getFirst().get(i).getWeight(j)));
                }
                writer.close();
            }
            for (int i = 1; i < hiddenLayersSizes.length; i++) {
                Files.createDirectories(Paths.get(directoryPath + "/neural_networks/network" + networkNo + "/layers/layer" + i));

                for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                    BufferedWriter writer = new BufferedWriter(new FileWriter(directoryPath + "/neural_networks/network" + networkNo + "/layers/layer" + i + "/neuron" + j + ".txt"));
                    writer.write(String.valueOf(hiddenLayers.get(i).get(j).getBias()));
                    for (int k = 0; k < hiddenLayersSizes[i-1]; k++) {
                        writer.newLine();
                        writer.write(String.valueOf(hiddenLayers.get(i).get(j).getWeight(k)));
                    }
                    writer.close();
                }
            }
            Files.createDirectories(Paths.get(directoryPath + "/neural_networks/network" + networkNo + "/layers/layer" + hiddenLayersSizes.length));
            for (int i = 0; i < outputLayerSize; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(directoryPath + "/neural_networks/network" + networkNo + "/layers/layer"
                        + hiddenLayersSizes.length + "/neuron" + i + ".txt"));
                writer.write(String.valueOf(outputLayer.get(i).getBias()));
                for (int j = 0; j < outputLayer.get(i).getWeights().length; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(outputLayer.get(i).getWeight(j)));
                }
                writer.close();
            }
        }catch (IOException e){
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    /**
     * Sigmoid function which changes any number to number between 1 and 0. The function is not linear, and it is really steep around zero.
     * @param x the number you want to convert.
     * @return value of the x calculated with the sigmoid function.
     */
    private double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
}