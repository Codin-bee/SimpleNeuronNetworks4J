package com.codingbee.snn4j.objects_for_parsing;

public class ExampleJsonTwo {
    double[] expectedResults;
    double[] values;

    public ExampleJsonTwo(double[] expectedResults, double[] values) {
        this.expectedResults = expectedResults;
        this.values = values;
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
