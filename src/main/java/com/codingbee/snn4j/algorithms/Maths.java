package com.codingbee.snn4j.algorithms;

public class Maths {
    public static float[] multiplyWbyV(float[][] matrix, float[] vector){
        int rows = matrix.length;
        int cols = matrix[0].length;

        float[] outputVector = new float[rows];

        for (int i = 0; i < rows; i++) {
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            outputVector[i] = sum;
        }

        return outputVector;
    }

    public static float[] multiplyVectors(float[] vectorA, float[] vectorB){
        float[] product = new float[vectorA.length];
        for (int i = 0; i < vectorA.length; i++) {
            product[i] = vectorA[i] * vectorB[i];
        }
        return product;
    }

    public static float[] addVectors(float[] vectorA, float[] vectorB){
        float[] sum = new float[vectorA.length];
        for (int i = 0; i < vectorA.length; i++) {
            sum[i] = vectorA[i] + vectorB[i];
        }
        return sum;
    }

    public static float[] concatenateVectors(float[] vectorA, float[] vectorB){
        float[] concatenatedVector = new float[vectorA.length + vectorB.length];
        System.arraycopy(vectorB, 0, concatenatedVector, 0, vectorB.length);
        System.arraycopy(vectorA, 0, concatenatedVector, vectorB.length, vectorA.length);
        return concatenatedVector;
    }

    public static int getIndexWithHighestVal(float[] array){
        float highestVal = array[0];
        int index = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > highestVal){
                highestVal = array[i];
                index = i;
            }
        }
        return index;
    }

    public static float[][][] allocateArrayOfSameSize(float[][][] reference){
        float[][][] newArray = new float[reference.length][][];
        for (int i = 0; i < reference.length; i++) {
            newArray[i] = new float[reference[i].length][];
            for (int j = 0; j < reference[i].length; j++) {
                newArray[i][j] = new float[reference[i][j].length];
            }
        }
        return newArray;
    }

    public static float[] multiplyTransposeWByV(float[][] matrix, float[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        float[] outputVector = new float[cols];

        for (int i = 0; i < cols; i++) {
            float sum = 0;
            for (int j = 0; j < rows; j++) {
                sum += matrix[j][i] * vector[j];
            }
            outputVector[i] = sum;
        }

        return outputVector;
    }

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
}
