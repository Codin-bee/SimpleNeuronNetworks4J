package com.codingbee.snn4j.algorithms;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

public class Algorithms {
    /**
     * Simple method used to find the element with the highest value.
     * Special case: If the highest value is found at more indexes returns the lowest of the indexes.
     * @param values array where the algorithm searches.
     * @return index of the element with the highest value
     * @throws IncorrectDataException in case the inserted array is null
     */
    public static int getIndexWithHighestVal(double[] values) throws IncorrectDataException {
        if(values == null){
            throw new IncorrectDataException("Get index with highest value - the array must not be null");
        }
        int indexWithHighestNo = 0;
        double highestNo = Double.MIN_VALUE;
        for (int i = 0; i < values.length; i++) {
            if (values[i]>highestNo){
                highestNo = values[i];
                indexWithHighestNo = i;
            }
        }
        return indexWithHighestNo;
    }
}
