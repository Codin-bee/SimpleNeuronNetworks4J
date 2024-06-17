package com.codingbee.snn4j.objects_for_parsing;

public class ExampleJsonOne {
    private int correctNeuronIndex;
    private double[] values;

    public ExampleJsonOne(int correctNumberIndex, double[] values) {
        this.correctNeuronIndex = correctNumberIndex;
        this.values = values;
    }

    //region Getters and setters without additional logic
    public int getCorrectNeuronIndex() {
        return correctNeuronIndex;
    }

    public void setCorrectNeuronIndex(int correctNeuronIndex) {
        this.correctNeuronIndex = correctNeuronIndex;
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        this.values = values;
    }
    //endregion
}
