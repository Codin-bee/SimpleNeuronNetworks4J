package com.codingbee.snn4j.neural_networks.mlp;

import com.codingbee.snn4j.exceptions.IncorrectDataException;

@SuppressWarnings("unused")
public class MLPCalculator {
    public static int getNumberOfParams(int[] layerLengths){
        if (layerLengths == null){
            throw new IncorrectDataException("Given array can not be null");
        }
        if (layerLengths.length <= 1){
            throw new IncorrectDataException("The MLP has to contain at least two layers");
        }
        if(layerLengths[0] < 0){
            throw new IncorrectDataException("Length of layer can not be a negative number");
        }
        int numberOfParams = 0;
        for (int i = 1; i < layerLengths.length; i++){
            if(layerLengths[i] < 0){
                throw new IncorrectDataException("Length of layer can not be a negative number");
            }
            numberOfParams += layerLengths[i] * layerLengths[i-1];
        }
        return numberOfParams;
    }
}