package com.codingbee.neural_network.neural_network;

public class Neuron {
    public static final double LAST = 0;
    private double currentValue;
    private final double bias;
    private final double[] weights;

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public void processNums(double[] nums){
        for (int i = 0; i < nums.length; i++) {
            currentValue += nums[i] * weights[i];
        }
    }

    /**
     * Returns final value and resets the neuron to its initial state.
     * @return final value of the neuron. Calculated from Neurons current value and bias by passing their difference through sigmoid and ReLU functions.
     */
    public double getFinalValue(){
        double finalValue = sigmoid(ReLU(currentValue - bias));
        currentValue = 0;
        return finalValue;
    }

    /**
     * Sigmoid function which changes any number to number between 1 and -1. The function is not linear, but it is really steep around zero.
     * @param x the number you want to convert.
     * @return value of the x calculated with the sigmoid function.
     */
    private double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    /**
     *ReLU function which returns the same number if it is positive and if it is negative returns zero.
     * @param x the number you want to convert.
     * @return value of the x calculated with the ReLU function.
     */
    private double ReLU(double x){
        if(x < 0) return 0;
        return x;
    }

    //region Getters and setters without additional logic

    public double getCurrentValue() {
        return currentValue;
    }

    public void setCurrentValue(double currentValue) {
        this.currentValue = currentValue;
    }

    public double getBias() {
        return bias;
    }

    public double[] getWeights() {
        return weights;
    }

    //endregion

}
