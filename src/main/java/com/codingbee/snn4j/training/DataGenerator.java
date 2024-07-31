package com.codingbee.snn4j.training;

import com.codingbee.snn4j.algorithms.AlgorithmManager;
import com.codingbee.snn4j.enums.DataFormat;
import com.codingbee.snn4j.exceptions.DevelopmentException;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.objects_for_parsing.JsonOne;
import com.codingbee.snn4j.objects_for_parsing.JsonTwo;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

public class DataGenerator {
    public void saveIntoDir(String dirPath, DataFormat format, Dataset data) throws FileManagingException {
        switch (format){
            case DataFormat.JSON_ONE -> {
                ObjectMapper mapper = new ObjectMapper();
                AlgorithmManager algorithmManager = new AlgorithmManager();
                for (int i = 0; i < data.getInputData().length; i++) {
                    JsonOne example = new JsonOne(algorithmManager.getIndexWithHighestVal(data.getInputData()[i]), data.getExpectedResults()[i]);
                    try {
                        mapper.writeValue(new File(dirPath + "/example" + i), example);
                    } catch (IOException e) {
                        throw new FileManagingException(e.getLocalizedMessage());
                    }
                }
            }
            case DataFormat.JSON_TWO -> {
                ObjectMapper mapper = new ObjectMapper();
                for (int i = 0; i < data.getInputData().length; i++) {
                    JsonTwo example = new JsonTwo(data.getInputData()[i], data.getExpectedResults()[i]);
                    try {
                        mapper.writeValue(new File(dirPath + "/example" + i), example);
                    } catch (IOException e) {
                        throw new FileManagingException(e.getLocalizedMessage());
                    }
                }
            }
            default -> throw new DevelopmentException("Data format not implemented yet.");
        }
    }
}
