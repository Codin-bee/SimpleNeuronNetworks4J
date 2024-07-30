package com.codingbee.snn4j.objects_for_parsing;

public class JsonTwo {
    double[] expectedResults;
    double[] values;

    @SuppressWarnings("unused")
    public JsonTwo(double[] expectedResults, double[] values) {
        this.expectedResults = expectedResults;
        this.values = values;
    }

    @SuppressWarnings("unused")
    public JsonTwo(){

    }

    @SuppressWarnings("unused")
    public double[] getExpectedResults() {
        return expectedResults;
    }

    @SuppressWarnings("unused")
    public void setExpectedResults(double[] expectedResults) {
        this.expectedResults = expectedResults;
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
