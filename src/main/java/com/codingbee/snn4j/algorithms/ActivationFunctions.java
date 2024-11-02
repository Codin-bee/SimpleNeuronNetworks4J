package com.codingbee.snn4j.algorithms;

public class ActivationFunctions {

    public static void softmaxInPlace(double[] values, double temp){
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.exp(values[i] / temp);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }

    public static double leakyReLU(double x, double alpha){
        if(x < 0){
            return 0;
        }
        return x * alpha;
    }

}
