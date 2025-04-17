package com.codingbee.snn4j.algorithms;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

public class Maths {
    /**
     * Multiplies matrix by a vector and returns new vector as a result
     * @param matrix the matrix to be multiplied
     * @param vector vector of the same length as matrix columns
     * @return new vector as a product of the multiplication
     * @throws IncorrectDataException if any null values are passed or the matrix and vector can not be multiplied
     */
    public static float[] multiplyWbyV(float[][] matrix, float[] vector){
        int rows = matrix.length;
        int cols = matrix[0].length;

        if (vector.length != cols){
            throw new IncorrectDataException("The matrix and vector are incompatible for multiplication");
        }

        float[] outputVector = new float[rows];

        for (int i = 0; i < rows; i++) {
            if (matrix[i] == null){
                throw new IncorrectDataException("The passed argument can not be a null");
            }
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            outputVector[i] = sum;
        }

        return outputVector;
    }

    /**
     * Multiplies two vectors element-wise. Returns their product vector.
     * @param vectorA first vector
     * @param vectorB second vector
     * @return new vector of the same length
     * @throws IncorrectDataException if any vector is null or their length does not match
     */
    public static float[] multiplyVectors(float[] vectorA, float[] vectorB){
        if (vectorA == null || vectorB == null){
            throw new IncorrectDataException("The passed argument can not be a null.");
        }
        if (vectorA.length != vectorB.length){
            throw new IncorrectDataException("The lengths of two arrays has to be the same in order to multiply them element-wise.");
        }
        float[] product = new float[vectorA.length];
        for (int i = 0; i < vectorA.length; i++) {
            product[i] = vectorA[i] * vectorB[i];
        }
        return product;
    }

    /**
     * Adds two vectors element-wise. Returns their sum vector.
     * @param vectorA first vector
     * @param vectorB second vector
     * @return new vector of the same length
     * @throws IncorrectDataException if any array is null or their length does not match
     */
    public static float[] addVectors(float[] vectorA, float[] vectorB){
        if (vectorA == null || vectorB == null){
            throw new IncorrectDataException("The passed argument can not be a null.");
        }
        if (vectorA.length != vectorB.length){
            throw new IncorrectDataException("The lengths of two arrays has to be the same in order to add them element-wise.");
        }
        float[] sum = new float[vectorA.length];
        for (int i = 0; i < vectorA.length; i++) {
            sum[i] = vectorA[i] + vectorB[i];
        }
        return sum;
    }

    /**
     * Concatenates two vectors and returns a new one with the combined length of the two vectors.
     * @param vectorA first vector, starts at index 0 in the new array
     * @param vectorB second vector, starts after the previous one in the new array
     * @return new vector with combined length of vectorA and vectorB, containing all their elements in order
     * @throws IncorrectDataException if a null is passed to the method
     */
    public static float[] concatVectors(float[] vectorA, float[] vectorB){
        if (vectorA == null || vectorB == null){
            throw new IncorrectDataException("The passed argument can not be a null.");
        }
        float[] concatenatedVector = new float[vectorA.length + vectorB.length];
        System.arraycopy(vectorB, 0, concatenatedVector, 0, vectorB.length);
        System.arraycopy(vectorA, 0, concatenatedVector, vectorB.length, vectorA.length);
        return concatenatedVector;
    }

    /**
     * Finds the element with the largest value and returns its index in the array.
     * @param array the array to search through
     * @return index of the largest element
     * @throws IncorrectDataException if null array is passed to the method
     */
    public static int getIndexWithHighestVal(float[] array){
        if (array == null){
            throw new IncorrectDataException("The passed argument can not be a null.");
        }
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

    /**
     * Allocates a new 3-dimensional array of the same dimensions as the reference array
     * @param reference the array with desired dimensions
     * @return newly allocated 3-dimensional array with same dimensions as the passed reference array
     * @throws IncorrectDataException if there are any null values in the reference array
     */
    public static float[][][] allocateArrayOfSameSize(float[][][] reference){
        if (reference == null){
            throw new IncorrectDataException("The passed argument can not be a null");
        }
        float[][][] newArray = new float[reference.length][][];
        for (int i = 0; i < reference.length; i++) {
            if (reference[i] == null){
                throw new IncorrectDataException("The passed argument can not be a null");
            }
            newArray[i] = new float[reference[i].length][];
            for (int j = 0; j < reference[i].length; j++) {
                if (reference[i][j] == null){
                    throw new IncorrectDataException("The passed argument can not be a null");
                }
                newArray[i][j] = new float[reference[i][j].length];
            }
        }
        return newArray;
    }

    /**
     * Multiplies transpose of the matrix by the vector and return their product vector
     * @param matrix the matrix
     * @param vector the vector with same length as number of vector rows
     * @return new vector, the product of the multiplication
     * @throws IncorrectDataException if any null values are passed or the matrix and vector can not be multiplied
     */
    public static float[] multiplyTransposeWByV(float[][] matrix, float[] vector) {
        if (matrix == null || vector == null || matrix[0] == null){
            throw new IncorrectDataException("The passed argument can not be a null");
        }
        int rows = matrix.length;
        int cols = matrix[0].length;

        if (vector.length != rows){
            throw new IncorrectDataException("The transpose of the matrix and the vector are incompatible for multiplication");
        }

        float[] outputVector = new float[cols];

        for (int i = 0; i < cols; i++) {
            float sum = 0;
            for (int j = 0; j < rows; j++) {
                if (matrix[i] == null){
                    throw new IncorrectDataException("The passed argument can not be a null");
                }
                sum += matrix[j][i] * vector[j];
            }
            outputVector[i] = sum;
        }

        return outputVector;
    }
}
