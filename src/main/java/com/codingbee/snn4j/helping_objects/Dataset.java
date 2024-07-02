package com.codingbee.snn4j.helping_objects;

public class Dataset {
    private double[][] inputData;
    private double[][] expectedResults;

    /**
     * Object which holds data, used for training or testing the network
     * @param inputData the values inserted into input array
     * @param expectedResults the values expected in the output layer
     */
    public Dataset(double[][] inputData, double[][] expectedResults) {
        this.inputData = inputData;
        this.expectedResults = expectedResults;
    }

    public Dataset(){
    }

    @SuppressWarnings("unused")
    public double[][] getInputData() {
        return inputData;
    }

    @SuppressWarnings("unused")
    public void setInputData(double[][] inputData) {
        this.inputData = inputData;
    }

    @SuppressWarnings("unused")
    public double[][] getExpectedResults() {
        return expectedResults;
    }

    @SuppressWarnings("unused")
    public void setExpectedResults(double[][] expectedResults) {
        this.expectedResults = expectedResults;
    }
}