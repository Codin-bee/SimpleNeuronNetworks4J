package com.codingbee.snn4j.settings;

@SuppressWarnings("unused")
public class TrainingSettings {
    private float learningRate, exponentialDecayRateOne, exponentialDecayRateTwo, epsilon;
    private int batchSize;

    public TrainingSettings(float learningRate, float exponentialDecayRateOne, float exponentialDecayRateTwo, float epsilon, int batchSize) {
        this.learningRate = learningRate;
        this.exponentialDecayRateOne = exponentialDecayRateOne;
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
        this.epsilon = epsilon;
        this.batchSize = batchSize;
    }

    public TrainingSettings() {
        this(0.001f, 0.9f, 0.999f, 1e-8f, 64);
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public void setExponentialDecayRateOne(float exponentialDecayRateOne) {
        this.exponentialDecayRateOne = exponentialDecayRateOne;
    }

    public void setExponentialDecayRateTwo(float exponentialDecayRateTwo) {
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    //region GETTERS

    public float getLearningRate() {
        return learningRate;
    }

    public float getExponentialDecayRateOne() {
        return exponentialDecayRateOne;
    }

    public float getExponentialDecayRateTwo() {
        return exponentialDecayRateTwo;
    }

    public float getEpsilon() {
        return epsilon;
    }

    public int getBatchSize() {
        return batchSize;
    }

    //endregion
}
