package com.codingbee.neural_network.objects_for_parsing;

public class TrainingExample {
    private int correctNumber;
    private double[] weights;

    public TrainingExample() {
    }

    public TrainingExample(int correctNumber, double[] weights) {
        this.correctNumber = correctNumber;
        this.weights = weights;
    }

    public int getCorrectNumber() {
        return correctNumber;
    }

    public void setCorrectNumber(int correctNumber) {
        this.correctNumber = correctNumber;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }
}
