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

    //region Getters and setters without additional logic
    public double[][] getInputData() {
        return inputData;
    }

    public void setInputData(double[][] inputData) {
        this.inputData = inputData;
    }

    public double[][] getExpectedResults() {
        return expectedResults;
    }

    public void setExpectedResults(double[][] expectedResults) {
        this.expectedResults = expectedResults;
    }
    //endregion
}