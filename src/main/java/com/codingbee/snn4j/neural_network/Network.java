package com.codingbee.snn4j.neural_network;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;

import java.util.*;

import com.fasterxml.jackson.databind.ObjectMapper;

import com.codingbee.snn4j.enums.ExampleDataFormat;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.objects_for_parsing.ExampleJsonOne;


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
    @SuppressWarnings("unused")
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
    @SuppressWarnings("unused")
    public void initNeuronsFromDir(String dirPath) throws FileManagingException {
        try {
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
     * @param values array of doubles showing how certain the network is, that the neuron on each index is the correct one
     * @return array same length as the output layer defined in constructor, each value will be between 0 and 1 depending on its
     * probability to be correct, which means: higher value, higher probability
     */
    public double[] processAsProbabilities(double[] values){
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
            values2[i] = outputLayer.get(i).getFinalValue();
        }
        return normalize(values2);
    }

    /**
     * Processes given values through the network and returns the networks decision about similarity with training data.
     * @param values array of doubles showing how certain the network is, that the neuron on each index is the correct one
     * @return the index of neuron with the largest probability to be correct
     */
    public int processAsIndex(double[] values){
        return getIndexWithHighestNo(processAsProbabilities(values));
    }

    /**
     * Trains the network to act more like you want based on the training examples. It corrects the values, but doesn't
     * overwrite the values in files so if you want to keep them you have to call {@link #saveNetworksValues saveNetworkValues()}
     * @param data data, which is the network trained on
     * @param learningRate double deciding how big changes should be done to the weights
     * @param printDebugInfo whether to print debug info like cost function or not
     */
    @SuppressWarnings("unused")
    public void train(Dataset data, double learningRate, boolean printDebugInfo){
        trainWeights(data, learningRate, printDebugInfo);
        trainBiases(data, learningRate, printDebugInfo);
    }

    private void trainWeights(Dataset data, double learningRate, boolean printDebugInfo){
        double[][] trainingDataSet = data.getInputData();
        double[][] expectedResults = data.getExpectedResults();

        if (printDebugInfo) {
            System.out.println("Starting cost: " + calculateAverageCost(trainingDataSet, expectedResults));
        }
        for (int i = 0; i < hiddenLayersSizes.length; i++) {
            for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                for (int k = 0; k < hiddenLayers.get(i).get(j).getWeights().length; k++) {
                    double originalCost = calculateAverageCost(trainingDataSet, expectedResults);
                    double originalWeight = hiddenLayers.get(i).get(j).getWeight(k);
                    hiddenLayers.get(i).get(j).setWeight(k, originalWeight + learningRate);
                    double costWithFirstNudge = calculateAverageCost(trainingDataSet, expectedResults);
                    if (originalCost < costWithFirstNudge) {
                        hiddenLayers.get(i).get(j).setWeight(k, originalWeight - learningRate);
                        double costWithRevertedNudge = calculateAverageCost(trainingDataSet, expectedResults);
                        if (costWithRevertedNudge > originalCost) {
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
                if (originalCost < costWithFirstNudge) {
                    outputLayer.get(i).setWeight(j, originalWeight - learningRate);
                    double costWithRevertedNudge = calculateAverageCost(trainingDataSet, expectedResults);
                    if (costWithRevertedNudge > originalCost) {
                        outputLayer.get(i).setWeight(j, originalWeight);
                    }
                }
            }
        }
        if (printDebugInfo) {
            System.out.println("Cost after training: " + calculateAverageCost(trainingDataSet, expectedResults));
    }
    }

    public void train2(Dataset data, int iterations, boolean printDebugInfo){
        double[][] trainingDataSet = data.getInputData();
        double[][] expectedResults = data.getExpectedResults();

        if (printDebugInfo) {
            System.out.println("Starting cost: " + calculateAverageCost(trainingDataSet, expectedResults));
        }
        double  alfa = 0.005,
                beta1 = 0.9,
                beta2 = 0.999,
                epsilon = 0.00000001,
                time = 0;
        int longestLayer = Math.max(Arrays.stream(hiddenLayersSizes).max().getAsInt(), outputLayerSize);

        double[][][]
                weightM = new double[hiddenLayers.size()+1][longestLayer][784],///HELLL NAAAH
                weightV = new double[hiddenLayers.size()+1][longestLayer][784];
        double[][]
                biasM = new double[hiddenLayers.size()+1][longestLayer],
                biasV = new double[hiddenLayers.size()+1][longestLayer];

        for (int i = 0; i < iterations; i++) {

            //WEIGHTS
            time++;
            for (int j = 0; j < hiddenLayersSizes.length; j++) {
                for (int k = 0; k < hiddenLayersSizes[j]; k++) {
                    for (int l = 0; l < hiddenLayers.get(j).get(k).getWeights().length; l++) {
                        double w = hiddenLayers.get(j).get(k).getWeight(l);
                        weightM[j][k][l] = beta1 * weightM[j][k][l] + (1 - beta1) * differentiateWeight(hiddenLayers.get(j).get(k), l, data);//del L/ del w
                        weightV[j][k][l] = beta2 * weightV[j][k][l] + (1 - beta2) * Math.pow((differentiateWeight(hiddenLayers.get(j).get(k), l, data)), 2);//del L/ del w
                        double m2 = weightM[j][k][l] / Math.pow((1 - beta1), time);
                        double v2 = weightV[j][k][l] / Math.pow((1 - beta2), time);
                        hiddenLayers.get(j).get(k).setWeight(l, w - m2 * (alfa / (Math.sqrt(weightV[j][k][l]) + epsilon)));
                    }
                }
            }
            for (int j = 0; j < outputLayerSize; j++) {
                for (int k = 0; k < hiddenLayersSizes[hiddenLayersSizes.length - 1]; k++) {
                    double w = outputLayer.get(j).getWeight(k);
                    weightM[hiddenLayers.size()][j][k] = beta1 * weightM[hiddenLayers.size()][j][k] + (1 - beta1) * differentiateWeight(outputLayer.get(j), k, data);//del L/ del w
                    weightV[hiddenLayers.size()][j][k] = beta2 * weightV[hiddenLayers.size()][j][k] + (1 - beta2) * Math.pow(differentiateWeight(outputLayer.get(j), k, data), 2);//del L/ del w
                    double m2 = weightM[hiddenLayers.size()][j][k] / Math.pow((1 - beta1), time);
                    double v2 = weightV[hiddenLayers.size()][j][k] / Math.pow((1 - beta2), time);
                    outputLayer.get(j).setWeight(k, w - m2 * (alfa / (Math.sqrt(v2)) + epsilon ));
                }
            }
            //BIASES
            for (int j = 0; j < hiddenLayersSizes.length; j++) {
                for (int k = 0; k < hiddenLayersSizes[j]; k++) {
                    double b = hiddenLayers.get(j).get(k).getBias();
                    biasM[j][k] = beta1 * biasM[j][k] + (1-beta1) * differentiateBias(hiddenLayers.get(j).get(k), data);
                    biasV[j][k] = beta2 * biasV[j][k] + (1-beta2) * Math.pow(differentiateBias(hiddenLayers.get(j).get(k), data),2);
                    double m2 = biasM[j][k] / Math.pow((1-beta1), time);
                    double v2 = biasV[j][k] / Math.pow((1-beta2), time);
                }

            }
            for (int j = 0; j < outputLayerSize; j++) {
                double b = outputLayer.get(j).getBias();
                biasM[hiddenLayers.size()][j] = beta1 * biasM[hiddenLayers.size()][j] + (1-beta1) * differentiateBias(outputLayer.get(j), data);
                biasV[hiddenLayers.size()][j] = beta2 * biasV[hiddenLayers.size()][j] + (1-beta2) * Math.pow(differentiateBias(outputLayer.get(j), data),2);
                double m2 = biasM[hiddenLayers.size()][j] / Math.pow((1-beta1), time);
                double v2 = biasV[hiddenLayers.size()][j] / Math.pow((1-beta2), time);
            }

        }
        if (printDebugInfo) {
            System.out.println("Cost after training: " + calculateAverageCost(trainingDataSet, expectedResults));
        }
    }

    private double differentiateWeight(Neuron neuron, int weightNo, Dataset data){
        double nudge = 0.005;
        double originalCost = calculateAverageCost(data.getInputData(), data.getExpectedResults());
        neuron.setWeight(weightNo, neuron.getWeight(weightNo) + nudge);
        double newCost = calculateAverageCost(data.getInputData(), data.getExpectedResults());
        neuron.setWeight(weightNo, neuron.getWeight(weightNo) - nudge);
        return (originalCost-newCost) / nudge ;
    }

    private double differentiateBias(Neuron neuron,  Dataset data){
        double nudge = 0.005;
        double originalCost = calculateAverageCost(data.getInputData(), data.getExpectedResults());
        neuron.setBias(neuron.getBias() + nudge);
        double newCost = calculateAverageCost(data.getInputData(), data.getExpectedResults());
        neuron.setBias(neuron.getBias() - nudge);
        return (originalCost-newCost) / nudge ;
    }

    private void trainBiases(Dataset data, double learningRate, boolean printDebugInfo){
        double[][] trainingDataSet = data.getInputData();
        double[][] expectedResults = data.getExpectedResults();
        if (printDebugInfo) {
            System.out.println("Starting cost: " + calculateAverageCost(trainingDataSet, expectedResults));
        }
        for (int i = 0; i < hiddenLayersSizes.length; i++) {
            for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                double originalCost = calculateAverageCost(trainingDataSet, expectedResults);
                double originalBias = hiddenLayers.get(i).get(j).getBias();
                hiddenLayers.get(i).get(j).setBias(originalBias + learningRate);
                double costWithFirstNudge = calculateAverageCost(trainingDataSet, expectedResults);
                if (originalCost < costWithFirstNudge) {
                    hiddenLayers.get(i).get(j).setBias(originalBias - learningRate);
                    double costWithRevertedNudge = calculateAverageCost(trainingDataSet, expectedResults);
                    if (costWithRevertedNudge > originalCost) {
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
        if (printDebugInfo) {
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
            double[] received = processAsProbabilities(trainingDataSet[i]);
            for (int j = 0; j < outputLayerSize; j++) {
                costsSummed += Math.pow(received[j] - expectedResults[i][j], 2);
            }
            numberOfCostsInSum++;
        }
        return costsSummed/numberOfCostsInSum;
    }

    /**
     * Loads training data into given object.
     * @param directoryPath path to directory with training data.
     * @param exampleDataFormat enum deciding how to read the files. More information in {@link ExampleDataFormat}
     * @param data object holding arrays with the data, this method writes values directly into the object
     * @throws FileManagingException if some problem arises while working with files
     */
    public void loadData(String directoryPath, ExampleDataFormat exampleDataFormat, Dataset data) throws FileManagingException {
        double[][] trainingDataSet;
        double[][] expectedResults;
        try {
            switch (exampleDataFormat) {
                case JSON_ONE -> {
                    ObjectMapper mapper = new ObjectMapper();
                    File directory = new File(directoryPath);
                    File[] files = directory.listFiles((dir, name) -> name.endsWith("json"));
                    if (files != null) {
                        trainingDataSet = new double[files.length][inputLayerSize];
                        expectedResults = new double[files.length][outputLayerSize];
                        for (int i = 0; i < files.length; i++) {
                            Arrays.fill(expectedResults[i], 0);
                            ExampleJsonOne example = mapper.readValue(files[i], ExampleJsonOne.class);
                            trainingDataSet[i] = example.getValues();
                            expectedResults[i][example.getCorrectNeuronIndex()] = 1;
                        }

                        data.setInputData(trainingDataSet);
                        data.setExpectedResults(expectedResults);
                    }
                }
                case JSON_TWO -> {

                }
                default -> throw new IncorrectDataException("Data format not implemented yet.");
            }
        } catch (IncorrectDataException e) {
            throw new IncorrectDataException(e.getLocalizedMessage());
        } catch (IOException e) {
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    /**
     * Saves networks values back to files.
     * @param directoryPath path to the files
     * @throws FileManagingException if some problem arises while working with files
     */
    @SuppressWarnings("unused")
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
    @SuppressWarnings("unused")
    public double getCorrectPercentage(String directoryPath, ExampleDataFormat exampleDataFormat) throws FileManagingException {
        Dataset testingData = new Dataset(null, null);
        loadData(directoryPath, exampleDataFormat, testingData);
        return getCorrectPercentage(testingData);
        }

    public double getCorrectPercentage(Dataset testingData){
        int correct = 0, total = 0;
        for (int i = 0; i < testingData.getInputData().length; i++) {
            total++;
            if (testingData.getExpectedResults()[i][processAsIndex(testingData.getInputData()[i])] == 1) correct ++;
        }
        return (double) correct/ (double) total * 100.0;
    }

    private int getIndexWithHighestNo(double[] nums){
        int indexWithHighestNo = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i]>indexWithHighestNo) indexWithHighestNo = i;
        }
        return indexWithHighestNo;
    }

    private double[] normalize(double[] nums){
        Arrays.sort(nums);
        double min = nums[0], max = nums[nums.length-1], diff = max-min;
        for (int i = 0; i < nums.length; i++) {
            nums[i] = (nums[i]-min) / diff;
        }
        return nums;
    }
    }