package com.codingbee.snn4j.objects_for_parsing;

public class ExampleJsonTwo {
    double[] expectedResults;
    double[] values;

    public ExampleJsonTwo(double[] expectedResults, double[] values) {
        this.expectedResults = expectedResults;
        this.values = values;
    }

    //region Getters and setters without additional logic

    public double[] getExpectedResults() {
        return expectedResults;
    }

    public void setExpectedResults(double[] expectedResults) {
        this.expectedResults = expectedResults;
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        this.values = values;
    }

    //endregion
}
