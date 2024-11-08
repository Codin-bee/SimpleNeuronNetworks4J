package com.codingbee.snn4j.helping_objects.objects_for_parsing;

public class JsonOne {
    private int correctNeuronIndex;
    private double[] values;

    @SuppressWarnings("unused")
    public JsonOne(int correctNeuronIndex, double[] values) {
        this.correctNeuronIndex = correctNeuronIndex;
        this.values = values;
    }

    @SuppressWarnings("unused")
    public JsonOne(){
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
