package com.codingbee.neural_network;

import com.codingbee.exceptions.IncorrectDataException;

import java.util.ArrayList;
import java.util.List;

public class Network {
    private List<List<Neuron>> hiddenLayers;
    private List<Neuron> outputLayer;
    private int[] hiddenLayersSizes;
    private int inputLayerSize, outputLayerSize;

    /**Creates new Network object based on the parameters it is given.
     *
     * @param inputLayerSize Amount of parameters passed to the input layer of the network. Must be higher than 0.
     * @param outputLayerSize The number of outcomes in the output layer. Must be higher than 0.
     * @param hiddenLayersSizes Array of numbers which determine how many neurons will each hidden layer have. If it is null no hidden layer will be crated.
     *
     * @throws IncorrectDataException If requirements of the parameters are not fulfilled.
     */
    public Network(int inputLayerSize, int outputLayerSize, int[] hiddenLayersSizes) throws IncorrectDataException {
        if (inputLayerSize<1) throw new IncorrectDataException("Network constructor - input layer size");
        if (outputLayerSize<1) throw new IncorrectDataException("Network constructor - output layer size");
        if (hiddenLayersSizes!=null) for (int hiddenLayerSize : hiddenLayersSizes) {
            if (hiddenLayerSize < 1) throw new IncorrectDataException("Network constructor - hidden layers sizes");
        }
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.hiddenLayersSizes = hiddenLayersSizes;
        hiddenLayers = new ArrayList<>();
        outputLayer = new ArrayList<>();
    }
}