package com.codingbee.neural_network.neural_network;

import com.codingbee.neural_network.exceptions.FileManagingException;
import com.codingbee.neural_network.exceptions.IncorrectDataException;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.FileWriter;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

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
            Files.createDirectories(Paths.get(dirPath + "/neural_networks/network" + networkNo + "/layers/layer0"));
            for (int i = 0; i < hiddenLayersSizes[0]; i++) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + "/neural_networks/network" + networkNo + "/layers/layer0/neuron" + i + ".txt"));
                writer.write(String.valueOf(Math.floor(Math.random()*10000)/10000));
                for (int j = 0; j < inputLayerSize; j++) {
                    writer.newLine();
                    writer.write(String.valueOf(Math.floor(Math.random()*10000)/10000));
                }
                writer.close();
            }
            for (int i = 1; i < hiddenLayersSizes.length; i++) {
                Files.createDirectories(Paths.get(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + i));

                for (int j = 0; j < hiddenLayersSizes[i]; j++) {
                    BufferedWriter writer = new BufferedWriter(new FileWriter(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + i + "/neuron" + j + ".txt"));
                    writer.write(String.valueOf(Math.floor(Math.random()*10000)/10000));
                    for (int k = 0; k < hiddenLayersSizes[i-1]; k++) {
                        writer.newLine();
                        writer.write(String.valueOf(Math.floor(Math.random()*10000)/10000));
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
                    writer.write(String.valueOf(Math.floor(Math.random()*10000)/10000));
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

            for (int i = 0; i < hiddenLayersSizes.length; i++) {
                List<Neuron> tempNeurons = new ArrayList<>();
                for (int j = 0; j < hiddenLayersSizes[0]; j++) {
                    BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + i
                        + "/neuron" + j + ".txt"));
                    double bias = Double.parseDouble(reader.readLine());
                    double[] weights = new double[hiddenLayersSizes.length];
                    for (int k = 0; k < hiddenLayersSizes[i]; k++) {
                        weights[k] = Double.parseDouble(reader.readLine());
                    }
                    tempNeurons.add(new Neuron(weights, bias));
                }
                hiddenLayers.add(tempNeurons);
            }

            for (int i = 0; i < outputLayerSize; i++) {

                BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/neural_networks/network" + networkNo + "/layers/layer" + hiddenLayersSizes.length
                        + "/neuron" + i + ".txt"));
                double bias = Double.parseDouble(reader.readLine());
                double[] weights = new double[hiddenLayersSizes.length];
                for (int k = 0; k < hiddenLayersSizes[i]; k++) {
                    weights[k] = Double.parseDouble(reader.readLine());
                }
                outputLayer.add(new Neuron(weights, bias));
            }
        }catch (IOException e){
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    /**
     * Process given values through the network and returns the networks decision about similarity with training data.
     * @param values array of doubles which represent some variables depending on your application
     * @return array same length as the output layer defined in constructor, each value will be between 0 and 1 depending on its
     * probability to be correct, which means higher value, higher probability
     */
    public double[] process(double[] values){
        double[] values2;
        for (int i = 0; i < hiddenLayers.size(); i++) {
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
        return values2;
    }
}