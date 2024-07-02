package com.codingbee.snn4j.objects_for_parsing;

public class ExampleJsonOne {
    private int correctNeuronIndex;
    private double[] values;

    public ExampleJsonOne(int correctNumberIndex, double[] values) {
        this.correctNeuronIndex = correctNumberIndex;
        this.values = values;
    }

    @SuppressWarnings("unused")
    public int getCorrectNeuronIndex() {
        return correctNeuronIndex;
    }

    @SuppressWarnings("unused")
    public void setCorrectNeuronIndex(int correctNeuronIndex) {
        this.correctNeuronIndex = correctNeuronIndex;
    }

    @SuppressWarnings("unused")
    public double[] getValues() {
        return values;
    }

    @SuppressWarnings("unused")
    public void setValues(double[] values) {
        this.values = values;
    }
}
