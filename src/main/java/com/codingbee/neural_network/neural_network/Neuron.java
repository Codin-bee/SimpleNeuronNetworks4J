package com.codingbee.neural_network.neural_network;

public class Neuron {
    public static final double LAST = 0;
    private double currentValue;
    private double bias;
    private double[] weights;

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
     * @return final value of the neuron. Calculated from Neurons current value and bias by passing their sum through ReLU function.
     */
    public double getFinalValue(){
        double finalValue = ReLU(currentValue + bias);
        currentValue = 0;
        return finalValue;
    }

    /**
     *ReLU function which returns the same number if it is positive and if it is negative returns zero.
     * @param x the number you want to convert.
     * @return value of the x calculated with the ReLU function.
     */
    private double ReLU(double x){
        System.err.println("relu " + x);
        if(x < 0){
            return 0;
        }
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

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    //endregion

    public double getWeight(int index){
        return weights[index];
    }

    public void setWeight(int index, double weight){
        weights[index] = weight;
    }

}
