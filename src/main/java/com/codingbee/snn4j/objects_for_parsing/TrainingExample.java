package com.codingbee.snn4j.objects_for_parsing;

public class TrainingExample {
    private int correctNumber;
    private double[] values;

    public TrainingExample() {
    }

    public TrainingExample(int correctNumber, double[] values) {
        this.correctNumber = correctNumber;
        this.values = values;
    }

    public int getCorrectNumber() {
        return correctNumber;
    }

    public void setCorrectNumber(int correctNumber) {
        this.correctNumber = correctNumber;
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        this.values = values;
    }
}
