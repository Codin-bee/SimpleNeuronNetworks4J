package com.codingbee.snn4j.math_and_algorithms;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

@SuppressWarnings("unused")
public class Algorithms {
    //Double

    /**
     * Applies the softmax function to given array in-place instead of returning new one
     * @param values values to apply the softmax to
     * @param temp temperature variable, part of the formula
     */
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

    //Int
    /**
     * Simple method used to find the element with the highest value.
     * Special case: If the highest value is found at more indexes returns the lowest of the indexes.
     * @param values array where the algorithm searches.
     * @return index of the element with the highest value
     * @throws IncorrectDataException in case the inserted array is null
     */
    public static int getIndexWithHighestVal(int[] values) throws IncorrectDataException {
        if(values == null){
            throw new IncorrectDataException("Get index with highest value - the array must not be null");
        }
        int indexWithHighestNo = 0;
        int highestNo = values[0];
        for (int i = 0; i < values.length; i++) {
            if (values[i]>highestNo){
                highestNo = values[i];
                indexWithHighestNo = i;
            }
        }
        return indexWithHighestNo;
    }

    /**
     * Applies the softmax function to given array in-place instead of returning new one
     * @param values values to apply the softmax to
     * @param temp temperature variable, part of the formula
     */
    public static void softmaxInPlace(int[] values, int temp){
        int sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = (int) Math.exp((double) values[i] / temp);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }


    //Float
    /**
     * Simple method used to find the element with the highest value.
     * Special case: If the highest value is found at more indexes returns the lowest of the indexes.
     * @param values array where the algorithm searches.
     * @return index of the element with the highest value
     * @throws IncorrectDataException in case the inserted array is null
     */
    public static int getIndexWithHighestVal(float[] values) throws IncorrectDataException {
        if(values == null){
            throw new IncorrectDataException("Get index with highest value - the array must not be null");
        }
        int indexWithHighestNo = 0;
        float highestNo = values[0];
        for (int i = 0; i < values.length; i++) {
            if (values[i]>highestNo){
                highestNo = values[i];
                indexWithHighestNo = i;
            }
        }
        return indexWithHighestNo;
    }

    /**
     * Applies the softmax function to given array in-place instead of returning new one
     * @param values values to apply the softmax to
     * @param temp temperature variable, part of the formula
     */
    public static void softmaxInPlace(float[] values, float temp){
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.exp(values[i] / temp);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }


    //Long
    /**
     * Simple method used to find the element with the highest value.
     * Special case: If the highest value is found at more indexes returns the lowest of the indexes.
     * @param values array where the algorithm searches.
     * @return index of the element with the highest value
     * @throws IncorrectDataException in case the inserted array is null
     */
    public static int getIndexWithHighestVal(long[] values) throws IncorrectDataException {
        if(values == null){
            throw new IncorrectDataException("Get index with highest value - the array must not be null");
        }
        int indexWithHighestNo = 0;
        long highestNo = values[0];
        for (int i = 0; i < values.length; i++) {
            if (values[i]>highestNo){
                highestNo = values[i];
                indexWithHighestNo = i;
            }
        }
        return indexWithHighestNo;
    }

    /**
     * Applies the softmax function to given array in-place instead of returning new one
     * @param values values to apply the softmax to
     * @param temp temperature variable, part of the formula
     */
    public static void softmaxInPlace(long[] values, long temp){
        long sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = (long) Math.exp((double) values[i] / temp);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }


    //Short
    /**
     * Simple method used to find the element with the highest value.
     * Special case: If the highest value is found at more indexes returns the lowest of the indexes.
     * @param values array where the algorithm searches.
     * @return index of the element with the highest value
     * @throws IncorrectDataException in case the inserted array is null
     */
    public static int getIndexWithHighestVal(short[] values) throws IncorrectDataException {
        if(values == null){
            throw new IncorrectDataException("Get index with highest value - the array must not be null");
        }
        int indexWithHighestNo = 0;
        short highestNo = values[0];
        for (int i = 0; i < values.length; i++) {
            if (values[i]>highestNo){
                highestNo = values[i];
                indexWithHighestNo = i;
            }
        }
        return indexWithHighestNo;
    }

    /**
     * Applies the softmax function to given array in-place instead of returning new one
     * @param values values to apply the softmax to
     * @param temp temperature variable, part of the formula
     */
    public static void softmaxInPlace(short[] values, short temp){
        short sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = (short) Math.exp((double) values[i] / temp);
            sum += values[i];
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }
}