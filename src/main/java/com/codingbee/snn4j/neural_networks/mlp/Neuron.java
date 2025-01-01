package com.codingbee.snn4j.neural_networks.mlp;

import com.codingbee.tool_box.exceptions.IncorrectDataException;

public class Neuron {
    public static final double LAST = 0;
    private double currentValue;
    private double bias;
    private double[] weights;

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public void processNums(double[] nums) throws IncorrectDataException{
        if (nums.length > weights.length){
            throw new IncorrectDataException("Neuron - processing numbers - the array is longer than the amount of weights the network was initialized with");
        }
        for (int i = 0; i < nums.length; i++) {
            currentValue += nums[i] * weights[i];
        }
    }

    /**
     * Returns final value and resets the neuron to its initial state.
     * @return final value of the neuron. Calculated from Neurons current value and bias by passing their sum through ReLU function.
     */
    public double getFinalValue(){
        double finalValue = activate(currentValue + bias);
        currentValue = 0;
        return finalValue;
    }

    /**
     *Activation function, currently implemented: Leaky ReLU (a = 0.01)
     * @param x the number you want to convert.
     * @return value of the x calculated with this function.
     */
    private double activate(double x){
        if(x < 0){
            return x * 0.01;
        }
        return x;
    }

    @SuppressWarnings("unused")
    public double getBias() {
        return bias;
    }

    @SuppressWarnings("unused")
    public void setBias(double bias) {
        this.bias = bias;
    }

    @SuppressWarnings("unused")
    public double[] getWeights() {
        return weights;
    }

    @SuppressWarnings("unused")
    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    @SuppressWarnings("unused")
    public double getWeight(int index){
        return weights[index];
    }

    @SuppressWarnings("unused")
    public void setWeight(int index, double weight){
        weights[index] = weight;
    }

}