package com.codingbee.snn4j.neural_network;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


import com.codingbee.snn4j.algorithms.AlgorithmManager;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.exceptions.MethodCallingException;
import com.codingbee.snn4j.helping_objects.Dataset;


public class MLP {
    private final String networkName;
    private final List<List<Neuron>> hiddenLayers;
    private final List<Neuron> outputLayer;
    private final int[] hiddenLayersSizes;
    private final int inputLayerSize;
    private final int outputLayerSize;
    private boolean initialized = false;

    /**Creates new MLP(Multi-layer perceptron) based on the parameters it is given.
     *
     * @param inputLayerSize the number of neurons in the input layer. Must be higher than 0.
     * @param outputLayerSize the number of neurons in the output layer. Must be higher than 0.
     * @param hiddenLayersSizes array of numbers which determine how many neurons will each hidden layer have. If it is null no hidden layer will be created. Every value of array must be higher than zero.
     * @param MLPName name which the MLP will be assigned to in the file system. Needs to be unique or else the MLPs with the same name in file system will overwrite each other's values.Can not be an empty String("").
     *
     * @throws IncorrectDataException if requirements of any parameter are not fulfilled.
     */
    public MLP(int inputLayerSize, int outputLayerSize, int[] hiddenLayersSizes, String MLPName) throws IncorrectDataException {
        if (inputLayerSize<1) throw new IncorrectDataException("MLP constructor - input layer size must be higher than zero");
        if (outputLayerSize<1) throw new IncorrectDataException("MLP constructor - output layer size must be higher than zero");
        if (hiddenLayersSizes!=null) {
            for (int hiddenLayerSize : hiddenLayersSizes) {
                if (hiddenLayerSize < 1) throw new IncorrectDataException("MLP constructor - all hidden layers sizes must be higher than zero");
            }
        }
        if (MLPName.isEmpty()) throw new IncorrectDataException("MLP constructor - MLP name must not be an empty String");

        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.hiddenLayersSizes = hiddenLayersSizes;
        this.networkName = MLPName;
        hiddenLayers = new ArrayList<>();
        outputLayer = new ArrayList<>();

    }

    /**
     * Generates random values for neuron initialization in given directory.
     * @param dirPath path where directories will be generated
     * @throws FileManagingException if some problem arises while working with files
     */
    @SuppressWarnings("unused")
    public void generateRandomNeuronsInDir(String dirPath) throws FileManagingException {
        try {
            Random rand = new Random();

            Files.createDirectories(Paths.get(dirPath + networkName + "/layers/layer0"));

            for (int i = 0; i < hiddenLayersSizes[0]; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + networkName + "/layers/layer0/neuron" + i + ".txt"));
                writer.write(String.valueOf(Math.random()));
                for (int j = 0; j < inputLayerSize; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(rand.nextGaussian() * Math.sqrt(2.0 / inputLayerSize)));
                }
                writer.close();
            }
            for (int i = 1; i < hiddenLayersSizes.length; i++) {
                Files.createDirectories(Paths.get(dirPath + networkName + "/layers/layer" + i));

                for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                    BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + networkName + "/layers/layer" + i + "/neuron" + j + ".txt"));
                    writer.write(String.valueOf(Math.random()));
                    for (int k = 0; k < hiddenLayersSizes[i-1]; k++) {
                        writer.newLine();
                        writer.write(String.valueOf(rand.nextGaussian() * Math.sqrt(2.0 / inputLayerSize)));
                    }
                    writer.close();
                }
            }
            Files.createDirectories(Paths.get(dirPath + networkName + "/layers/layer" + hiddenLayersSizes.length));
            for (int i = 0; i < outputLayerSize; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + networkName + "/layers/layer"
                        + hiddenLayersSizes.length + "/neuron" + i + ".txt"));
                writer.write(String.valueOf(Neuron.LAST));
                for (int j = 0; j < hiddenLayersSizes[hiddenLayersSizes.length-1]; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(rand.nextGaussian() * Math.sqrt(2.0 / inputLayerSize)));
                }
                writer.close();
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
                BufferedReader reader = new BufferedReader(new FileReader(dirPath + networkName + "/layers/layer0/neuron"
                        + j + ".txt"));
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
                    BufferedReader reader = new BufferedReader(new FileReader(dirPath + networkName + "/layers/layer" + i
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
                BufferedReader reader = new BufferedReader(new FileReader(dirPath + networkName + "/layers/layer" + hiddenLayersSizes.length + "/neuron" + i + ".txt"));
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
        initialized = true;
    }

    /**
     * Processes given values through the network and returns the networks decision about similarity with training data.
     * @param values array of doubles showing how certain the network is, that the neuron on each index is the correct one
     * @return array same length as the output layer defined in constructor, each value will be between 0 and 1 depending on its
     * probability to be correct, which means: higher value, higher probability
     */
    public double[] processAsProbabilities(double[] values) throws MethodCallingException {
        if(!initialized){
            throw new MethodCallingException("Can't process data - network hasn't been initialized yet");
        }
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
        return (values2);
    }

    /**
     * Processes given values through the network and returns the networks decision about similarity with training data.
     * @param values array of doubles showing how certain the network is, that the neuron on each index is the correct one
     * @return the index of neuron with the largest probability to be correct
     */
    public int processAsIndex(double[] values) throws MethodCallingException{
        return new AlgorithmManager().getIndexWithHighestVal(processAsProbabilities(values));
    }

    @SuppressWarnings("unused")
    public void train(Dataset data, int iterations, boolean printDebugInfo) throws MethodCallingException {
        if(!initialized){
            throw new MethodCallingException("Can't process data - network hasn't been initialized yet");
        }
        double[][] trainingDataSet = data.getInputData();
        double[][] expectedResults = data.getExpectedResults();

        if (printDebugInfo) {
            System.out.println("Starting cost: " + calculateAverageCost(trainingDataSet, expectedResults));
            System.out.println("Starting correct percentage: " + getCorrectPercentage(data));
        }
        double  alfa = 0.001,
                beta1 = 0.9,
                beta2 = 0.999,
                epsilon = 0.00000001,
                time = 0,
                mHat,
                vHat;
        int longestLayer = Math.max(Arrays.stream(hiddenLayersSizes).max().getAsInt(), outputLayerSize);

        double[][][]
                weightM = new double[hiddenLayers.size()+1][longestLayer][784],///HELLL NAAAH
                weightV = new double[hiddenLayers.size()+1][longestLayer][784];
        double[][]
                biasM = new double[hiddenLayers.size()+1][longestLayer],
                biasV = new double[hiddenLayers.size()+1][longestLayer];

        for (int i = 0; i < iterations; i++) {
            time++;

            //WEIGHTS
            for (int j = 0; j < hiddenLayersSizes.length; j++) {
                for (int k = 0; k < hiddenLayersSizes[j]; k++) {
                    for (int l = 0; l < hiddenLayers.get(j).get(k).getWeights().length; l++) {
                        double gradient = differentiateWeight(hiddenLayers.get(j).get(k), l, data);
                        weightM[j][k][l] = beta1 * weightM[j][k][l] + (1 - beta1) * gradient;//del L/ del w
                        weightV[j][k][l] = beta2 * weightV[j][k][l] + (1 - beta2) * Math.pow(gradient, 2);//del L/ del w
                        mHat = weightM[j][k][l] / (1 - Math.pow(beta1, time));
                        vHat = weightV[j][k][l] / (1 - Math.pow(beta2, time));
                        hiddenLayers.get(j).get(k).setWeight(l, hiddenLayers.get(j).get(k).getWeight(l) - mHat * (alfa / (Math.sqrt(vHat) + epsilon)));
                    }
                }
            }
            for (int j = 0; j < outputLayerSize; j++) {
                for (int k = 0; k < hiddenLayersSizes[hiddenLayersSizes.length - 1]; k++) {
                    weightM[hiddenLayers.size()][j][k] = beta1 * weightM[hiddenLayers.size()][j][k] + (1 - beta1) * differentiateWeight(outputLayer.get(j), k, data);//del L/ del w
                    weightV[hiddenLayers.size()][j][k] = beta2 * weightV[hiddenLayers.size()][j][k] + (1 - beta2) * Math.pow(differentiateWeight(outputLayer.get(j), k, data), 2);//del L/ del w
                    mHat = weightM[hiddenLayers.size()][j][k] / (1 - Math.pow(beta1, time));
                    vHat = weightV[hiddenLayers.size()][j][k] / (1 - Math.pow(beta2, time));
                    outputLayer.get(j).setWeight(k, outputLayer.get(j).getWeight(k) - mHat * (alfa / (Math.sqrt(vHat)) + epsilon ));
                }
            }
            //BIASES
            for (int j = 0; j < hiddenLayersSizes.length; j++) {
                for (int k = 0; k < hiddenLayersSizes[j]; k++) {
                    biasM[j][k] = beta1 * biasM[j][k] + (1-beta1) * differentiateBias(hiddenLayers.get(j).get(k), data);
                    biasV[j][k] = beta2 * biasV[j][k] + (1-beta2) * Math.pow(differentiateBias(hiddenLayers.get(j).get(k), data),2);
                    mHat = biasM[j][k] / (1 - Math.pow(beta1, time));
                    vHat = biasV[j][k] / (1 - Math.pow(beta2, time));
                    hiddenLayers.get(j).get(k).setBias(hiddenLayers.get(j).get(k).getBias() - mHat * ( alfa / (Math.sqrt(vHat) + epsilon )));
                }

            }
            for (int j = 0; j < outputLayerSize; j++) {
                biasM[hiddenLayers.size()][j] = beta1 * biasM[hiddenLayers.size()][j] + (1-beta1) * differentiateBias(outputLayer.get(j), data);
                biasV[hiddenLayers.size()][j] = beta2 * biasV[hiddenLayers.size()][j] + (1-beta2) * Math.pow(differentiateBias(outputLayer.get(j), data),2);
                mHat = biasM[hiddenLayers.size()][j] / (1 - Math.pow(beta1, time));
                vHat = biasV[hiddenLayers.size()][j] / (1 - Math.pow(beta2, time));
                outputLayer.get(j).setBias(outputLayer.get(j).getBias() - mHat * ( alfa / (Math.sqrt(vHat) + epsilon )));
            }
            if (printDebugInfo){
                System.out.println("Cost after " + (int) time + ". iteration: " + calculateAverageCost(trainingDataSet, expectedResults));
                System.out.println("Correct percentage after " + (int) time + ". iteration: " + getCorrectPercentage(data));
            }

        }
        if (printDebugInfo) {
            System.out.println("Cost after training: " + calculateAverageCost(trainingDataSet, expectedResults));
            System.out.println("Correct percentage after training: " + getCorrectPercentage(data));
        }
    }

    private double differentiateWeight(Neuron neuron, int weightNo, Dataset data) throws MethodCallingException {
        double nudge = 0.000001;
        double originalWeight = neuron.getWeight(weightNo);

        neuron.setWeight(weightNo, originalWeight + nudge);
        double costWithPositiveNudge = calculateAverageCost(data.getInputData(), data.getExpectedResults());

        neuron.setWeight(weightNo, originalWeight - nudge);
        double costWithNegativeNudge = calculateAverageCost(data.getInputData(), data.getExpectedResults());

        neuron.setWeight(weightNo, originalWeight);

        return (costWithPositiveNudge - costWithNegativeNudge) / (2 * nudge);
    }

    private double differentiateBias(Neuron neuron,  Dataset data) throws MethodCallingException {
        double nudge = 0.000001;
        double originalBias = neuron.getBias();

        neuron.setBias(originalBias + nudge);
        double costWithPositiveNudge = calculateAverageCost(data.getInputData(), data.getExpectedResults());

        neuron.setBias(originalBias - nudge);
        double costWithNegativeNudge = calculateAverageCost(data.getInputData(), data.getExpectedResults());

        neuron.setBias(originalBias);

        return (costWithPositiveNudge - costWithNegativeNudge) / (2 * nudge) ;
    }

    /**
     * Calculates cost function of given data.
     * @param trainingDataSet 2D array of training values
     * @param expectedResults 2D array of results you expect to get after processing the training data set
     * @return the average cost function
     */
    public double calculateAverageCost(double[][] trainingDataSet, double[][] expectedResults) throws MethodCallingException {
        if(!initialized){
            throw new MethodCallingException("Can't process data - network hasn't been initialized yet");
        }
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
     * Saves networks values to files.
     * @param directoryPath path to the files
     * @throws FileManagingException if some problem arises while working with files
     */
    @SuppressWarnings("unused")
    public void saveNetworksValues(String directoryPath) throws FileManagingException{
        try {
            Files.createDirectories(Paths.get(directoryPath + networkName + "/layers/layer0"));
            for (int i = 0; i < hiddenLayersSizes[0]; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(directoryPath +  networkName + "/layers/layer0/neuron" + i + ".txt"));
                writer.write(String.valueOf(hiddenLayers.getFirst().get(i).getBias()));
                for (int j = 0; j < inputLayerSize; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(hiddenLayers.getFirst().get(i).getWeight(j)));
                }
                writer.close();
            }
            for (int i = 1; i < hiddenLayersSizes.length; i++) {
                Files.createDirectories(Paths.get(directoryPath + networkName + "/layers/layer" + i));

                for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                    BufferedWriter writer = new BufferedWriter(new FileWriter(directoryPath + networkName + "/layers/layer" + i + "/neuron" + j + ".txt"));
                    writer.write(String.valueOf(hiddenLayers.get(i).get(j).getBias()));
                    for (int k = 0; k < hiddenLayersSizes[i-1]; k++) {
                        writer.newLine();
                        writer.write(String.valueOf(hiddenLayers.get(i).get(j).getWeight(k)));
                    }
                    writer.close();
                }
            }
            Files.createDirectories(Paths.get(directoryPath + networkName + "/layers/layer" + hiddenLayersSizes.length));
            for (int i = 0; i < outputLayerSize; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(directoryPath + networkName + "/layers/layer"
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
     * Returns how correct the network is on  the testing data as a percentage.
     * @param testingData the data u want to test the network on
     * @return percentage of how many answers the network got correct
     * @throws MethodCallingException if the method is called on network that hasn't been initialized
     */
    public double getCorrectPercentage(Dataset testingData) throws MethodCallingException {
        int correct = 0, total = 0;
        for (int i = 0; i < testingData.getInputData().length; i++) {
            total++;
            if (testingData.getExpectedResults()[i][processAsIndex(testingData.getInputData()[i])] == 1) correct ++;
        }
        return (double) correct/ (double) total * 100.0;
    }

    }