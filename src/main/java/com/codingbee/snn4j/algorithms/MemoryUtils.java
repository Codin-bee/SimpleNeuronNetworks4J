package com.codingbee.snn4j.algorithms;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

public class MemoryUtils {

    /**
     * Allocates a new 3-dimensional array of the same dimensions as the reference array
     * @param reference the array with desired dimensions
     * @return newly allocated 3-dimensional array with the same dimensions as the passed reference array
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
     * Allocates a new 2-dimensional array of the same dimensions as the reference array
     * @param reference the array with desired dimensions
     * @return newly allocated 2-dimensional array with the same dimensions as the passed reference array
     * @throws IncorrectDataException if there are any null values in the reference array
     */
    public static float[][] allocateArrayOfSameSize(float[][] reference){
        if (reference == null){
            throw new IncorrectDataException("The passed argument can not be a null");
        }
        float[][] newArray = new float[reference.length][];
        for (int i = 0; i < reference.length; i++) {
            if (reference[i] == null){
                throw new IncorrectDataException("The passed argument can not be a null");
            }
            newArray[i] = new float[reference[i].length];
        }
        return newArray;
    }

    /**
     * Validates whether the given 2-dimensional array is a matrix
     * @param matrix the matrix to validate
     * @throws IncorrectDataException if there are any null values in the matrix or the lengths of rows differ
     */
    public static void validateMatrix(float[][] matrix){
        if (matrix == null || matrix[0] == null){
            throw new IncorrectDataException("The passed argument cannot be a null");
        }
        int columns = matrix[0].length;

        for (float[] floats : matrix) {
            if (floats == null){
                throw new IncorrectDataException("The passed argument cannot be a null");
            }
            if (floats.length != columns) {
                throw new IncorrectDataException("The argument is not a proper matrix and contains arrays of varying sizes");
            }
        }
    }

    /**
     * Copies the given matrix and returns the copy
     * @param matrix the matrix to be copied
     * @return copy of the matrix
     * @throws IncorrectDataException if there are any null values in the matrix or the lengths of rows differ
     */
    public static float[][] copyMatrix(float[][] matrix){
        MemoryUtils.validateMatrix(matrix);
        float[][] copy = new float[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            copy[i] = new float[matrix[i].length];
            System.arraycopy(matrix[i], 0, copy[i], 0, matrix[i].length);
        }
        return copy;
    }
}
