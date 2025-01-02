package com.codingbee.snn4j.neural_networks;

@SuppressWarnings("unused")
public class TrainingSettings {
    private double learningRate, exponentialDecayRateOne, exponentialDecayRateTwo, epsilon;
    private int batchSize;

    public TrainingSettings(double learningRate, double exponentialDecayRateOne, double exponentialDecayRateTwo, double epsilon, int batchSize) {
        this.learningRate = learningRate;
        this.exponentialDecayRateOne = exponentialDecayRateOne;
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
        this.epsilon = epsilon;
        this.batchSize = batchSize;
    }

    public TrainingSettings() {
        this(0.001, 0.9, 0.999, 1e-8, 64);
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setExponentialDecayRateOne(double exponentialDecayRateOne) {
        this.exponentialDecayRateOne = exponentialDecayRateOne;
    }

    public void setExponentialDecayRateTwo(double exponentialDecayRateTwo) {
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    //region GETTERS

    public double getLearningRate() {
        return learningRate;
    }

    public double getExponentialDecayRateOne() {
        return exponentialDecayRateOne;
    }

    public double getExponentialDecayRateTwo() {
        return exponentialDecayRateTwo;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public int getBatchSize() {
        return batchSize;
    }

    //endregion
}
