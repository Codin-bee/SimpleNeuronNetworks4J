package com.codingbee.snn4j.algorithms;

public class AlgorithmManager {
    /**
     * Simple method used to find the element with the highest value.
     * @param nums array where the algorithm searches.
     * @return index of the element with the highest value
     */
    public int getIndexWithHighestNo(double[] nums){
        int indexWithHighestNo = 0;
        double highestNo = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i]>highestNo){
                highestNo = nums[i];
                indexWithHighestNo = i;
            }
        }
        return indexWithHighestNo;
    }
}
