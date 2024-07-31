package com.codingbee.snn4j.algorithms;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

public class AlgorithmManager {
    /**
     * Simple method used to find the element with the highest value.
     * Special case: If the highest value is found at more indexes returns the lowest of the indexes.
     * @param nums array where the algorithm searches.
     * @return index of the element with the highest value
     * @throws IncorrectDataException in case the inserted array is null
     */
    public int getIndexWithHighestVal(double[] nums) throws IncorrectDataException {
        if(nums == null){
            throw new IncorrectDataException("Get index with highest value - the array must not be null");
        }
        int indexWithHighestNo = 0;
        double highestNo = Double.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i]>highestNo){
                highestNo = nums[i];
                indexWithHighestNo = i;
            }
        }
        return indexWithHighestNo;
    }
}
