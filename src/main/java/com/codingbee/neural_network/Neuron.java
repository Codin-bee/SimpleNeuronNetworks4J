package com.codingbee.neural_network;

public class Neuron {
    public static final double LAST_NEURON = 0;
    private double currentValue, bias;
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
     * @return Final value of the neuron. Calculated with sigmoid and ReLU functions.
     */
    public double getFinalValue(){
        double finalValue = ReLU(sigmoid(currentValue + bias));
        currentValue = 0;
        return finalValue;
    }

    private double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    private double ReLU(double x){
        if(x < 0) return 0;
        return x;
    }

}
