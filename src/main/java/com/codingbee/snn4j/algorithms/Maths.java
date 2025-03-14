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
}
