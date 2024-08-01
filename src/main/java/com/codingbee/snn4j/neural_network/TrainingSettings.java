package com.codingbee.snn4j.neural_network;

public class TrainingSettings {
    private double learningRate, exponentialDecayRateOne, exponentialDecayRateTwo, epsilon;

    public TrainingSettings(double learningRate, double exponentialDecayRateOne, double exponentialDecayRateTwo, double epsilon) {
        this.learningRate = learningRate;
        this.exponentialDecayRateOne = exponentialDecayRateOne;
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
        this.epsilon = epsilon;
    }

    @SuppressWarnings("unused")
    public TrainingSettings() {
        this(0.001, 0.9, 0.999, 1e-8);
    }
    @SuppressWarnings("unused")
    public double getLearningRate() {
        return learningRate;
    }

    @SuppressWarnings("unused")
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @SuppressWarnings("unused")
    public double getExponentialDecayRateOne() {
        return exponentialDecayRateOne;
    }

    @SuppressWarnings("unused")
    public void setExponentialDecayRateOne(double exponentialDecayRateOne) {
        this.exponentialDecayRateOne = exponentialDecayRateOne;
    }

    @SuppressWarnings("unused")
    public double getExponentialDecayRateTwo() {
        return exponentialDecayRateTwo;
    }

    @SuppressWarnings("unused")
    public void setExponentialDecayRateTwo(double exponentialDecayRateTwo) {
        this.exponentialDecayRateTwo = exponentialDecayRateTwo;
    }

    @SuppressWarnings("unused")
    public double getEpsilon() {
        return epsilon;
    }

    @SuppressWarnings("unused")
    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
}
