package com.codingbee.snn4j.algorithms;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

public class Maths {
    /**
     * Multiplies matrix by a vector and returns new vector as a result
     * @param matrix the matrix to be multiplied
     * @param vector vector of the same length as matrix columns
     * @return new vector as a product of the multiplication
     * @throws IncorrectDataException if any null values are passed or the matrix and vector cannot be multiplied
     */
    public static float[] multiply(float[][] matrix, float[] vector){
        if (vector == null){
            throw new IncorrectDataException("The passed argument cannot be a null");
        }
        MemoryUtils.validateMatrix(matrix);

        int rows = matrix.length;
        int cols = matrix[0].length;

        if (vector.length != cols){
            throw new IncorrectDataException("The matrix and vector are incompatible for multiplication");
        }

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

    /**
     * Multiplies two vectors element-wise. Returns their product vector.
     * @param vectorA first vector
     * @param vectorB second vector
     * @return new vector of the same length
     * @throws IncorrectDataException if any vector is null or their length does not match
     */
    public static float[] multiplyElementWise(float[] vectorA, float[] vectorB){
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
        System.arraycopy(vectorA, 0, concatenatedVector, 0, vectorA.length);
        System.arraycopy(vectorB, 0, concatenatedVector, vectorA.length, vectorB.length);
        return concatenatedVector;
    }

    /**
     * Finds the element with the largest value and returns its index in the array.
     * @param array the array to search through
     * @return index of the largest element
     * @throws IncorrectDataException if a null or empty array is passed to the method
     */
    public static int getIndexOfLargestElement(float[] array){
        if (array == null){
            throw new IncorrectDataException("The passed argument can not be a null.");
        }
        if (array.length == 0){
            throw new IncorrectDataException("The passed array does not contain any values");
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
     * Multiplies transpose of the matrix by the vector and return their product vector
     * @param matrix the matrix
     * @param vector the vector with the same length as number of vector rows
     * @return new vector, the product of the multiplication
     * @throws IncorrectDataException if any null values are passed or the matrix and vector cannot be multiplied
     */
    public static float[] multiplyTranspose(float[][] matrix, float[] vector) {
        if (vector == null) {
            throw new IncorrectDataException("The passed argument cannot be a null");
        }

        MemoryUtils.validateMatrix(matrix);

        int rows = matrix.length;
        int cols = matrix[0].length;

        if (vector.length != rows) {
            throw new IncorrectDataException("The transpose of the matrix and the vector are incompatible for multiplication");
        }

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

    /**
     * Multiplies the two matrices using standard iterative algorithm
     * @param matrixA first matrix
     * @param matrixB second matrix
     * @return the product of the two matrices
     * @throws IncorrectDataException if any null values are passed or the matrices cannot be multiplied
     */
    public static float[][] multiply(float[][] matrixA, float[][] matrixB) {
        MemoryUtils.validateMatrix(matrixA);
        MemoryUtils.validateMatrix(matrixB);

        int rowsA = matrixA.length;
        int rowsB = matrixB.length;
        int columnsA = matrixA[0].length;
        int columnsB = matrixB[0].length;

        if (columnsA != rowsB){
            throw new IncorrectDataException("The matrices are unable to be multiplied");
        }

        float[][] result = new float[rowsA][columnsB];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < columnsB; j++) {
                for (int k = 0; k < columnsA; k++) {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }

        return result;
    }

    /**
     * Calculates the dot product of two vectors with the same length
     * @param vectorA first vector
     * @param vectorB second vector
     * @return the dot product of the two vectors
     * @throws IncorrectDataException if any null values are passed or the vector lengths do not match
     */
    public static float dotProduct(float[] vectorA, float[] vectorB){
        if (vectorA == null || vectorB == null){
            throw new IncorrectDataException("The passed argument cannot be a null");
        }
        if (vectorA.length != vectorB.length){
            throw new IncorrectDataException("The lengths of vectors have to be the same in order to calculate their dot product");
        }
        float sum = 0;
        for (int i = 0; i < vectorA.length; i++) {
            sum += vectorA[i] * vectorB[i];
        }
        return sum;
    }

    /**
     * Calculates the dyadic product of two vectors
     * @param vectorA first vector
     * @param vectorB second vector
     * @return newly created matrix, dyadic product of the vectors
     * @throws IncorrectDataException if any null values are passed
     */
    public static float[][] dyadicProduct(float[] vectorA, float[] vectorB){
        if (vectorA == null || vectorB == null){
            throw new IncorrectDataException("The passed argument cannot be a null");
        }
        float[][] dyadic = new float[vectorA.length][vectorB.length];

        for (int i = 0; i < vectorA.length; i++) {
            for (int j = 0; j < vectorB.length; j++) {
                dyadic[i][j] = vectorA[i] * vectorB[j];
            }
        }
        return dyadic;
    }

    /**
     * Adds elements of the second matrix to the first matrix in-place
     * @param matrixA first matrix, the result
     * @param matrixB second matrix, added to the first
     * @throws IncorrectDataException if any null values are passed
     */
    public static void addInPlace(float[][] matrixA, float[][] matrixB){
        MemoryUtils.validateMatrix(matrixA);
        MemoryUtils.validateMatrix(matrixB);

        if (matrixA.length != matrixB.length || matrixA[0].length != matrixB[0].length){
            throw new IncorrectDataException("The elements of given matrices can not be added because they have different dimensions");
        }

        for (int i = 0; i < matrixA.length; i++) {
            for (int j = 0; j < matrixA[i].length; j++) {
                matrixA[i][j] += matrixB[i][j];
            }
        }
    }

    /**
     * Scales every value of the matrix by given scalar in-place
     * @param matrix the matrix to be scaled
     * @param scale scalar by which the elements of the matrix are multiplied
     * @throws IncorrectDataException if any null values are passed
     */
    public static void scale(float[][] matrix, float scale){
        MemoryUtils.validateMatrix(matrix);

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] *= scale;
            }
        }
    }

}
