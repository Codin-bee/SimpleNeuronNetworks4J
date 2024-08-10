package com.codingbee.snn4j.helping_objects;

import com.codingbee.snn4j.algorithms.AlgorithmManager;
import com.codingbee.snn4j.enums.DataFormat;
import com.codingbee.snn4j.exceptions.DevelopmentException;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.objects_for_parsing.JsonOne;
import com.codingbee.snn4j.objects_for_parsing.JsonTwo;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Dataset {
    private double[][] inputData;
    private double[][] expectedResults;

    /**
     * Object which holds data, used for training or testing the network
     * @param inputData the values inserted into input array
     * @param expectedResults the values expected in the output layer
     */
    @SuppressWarnings("unused")
    public Dataset(double[][] inputData, double[][] expectedResults) {
        this.inputData = inputData;
        this.expectedResults = expectedResults;
    }

    @SuppressWarnings("unused")
    public Dataset(){
    }

    /**
     * Loads training data into the Dataset.
     * @param directoryPath path to directory with training data.
     * @param dataFormat enum deciding how to read the files. More information in {@link DataFormat}
     * @param networkOutputSize number of neurons in the output layer, must be initialized only when using JsonOne data format
     * @throws FileManagingException if some problem arises while working with files
     */
    @SuppressWarnings("unused")
    public void loadData(String directoryPath, DataFormat dataFormat, int networkOutputSize) throws FileManagingException {
        double[][] inputData;
        double[][] expectedResults;
        try {
            ObjectMapper mapper = new ObjectMapper();
            File directory = new File(directoryPath);
            File[] files = directory.listFiles((dir, name) -> name.endsWith("json"));
            if (files != null) {
                switch (dataFormat) {
                    case JSON_ONE -> {
                        inputData = new double[files.length][];
                        expectedResults = new double[files.length][networkOutputSize];
                        for (int i = 0; i < files.length; i++) {
                            Arrays.fill(expectedResults[i], 0);
                            JsonOne example = mapper.readValue(files[i], JsonOne.class);
                            inputData[i] = example.getValues();
                            expectedResults[i][example.getCorrectNeuronIndex()] = 1;
                        }

                    }
                    case JSON_TWO -> {
                        inputData = new double[files.length][];
                        expectedResults = new double[files.length][];
                        for (int i = 0; i < files.length; i++) {
                            JsonTwo example = mapper.readValue(files[i], JsonTwo.class);
                            inputData[i] = example.getValues();
                            expectedResults[i] = example.getExpectedResults();
                        }
                    }

                    default -> throw new DevelopmentException("Data format not implemented yet.");
                }
                this.setInputData(inputData);
                this.setExpectedResults(expectedResults);
            }
        } catch (IncorrectDataException e) {
            throw new IncorrectDataException(e.getLocalizedMessage());
        } catch (IOException e) {
            throw new FileManagingException(e.getLocalizedMessage());
        }
    }

    @SuppressWarnings("unused")
    public void saveIntoDir(String dirPath, DataFormat format) throws FileManagingException {
        switch (format){
            case DataFormat.JSON_ONE -> {
                ObjectMapper mapper = new ObjectMapper();
                AlgorithmManager algorithmManager = new AlgorithmManager();
                for (int i = 0; i < inputData.length; i++) {
                    JsonOne example = new JsonOne(algorithmManager.getIndexWithHighestVal(inputData[i]), expectedResults[i]);
                    try {
                        mapper.writeValue(new File(dirPath + "/example" + i), example);
                    } catch (IOException e) {
                        throw new FileManagingException(e.getLocalizedMessage());
                    }
                }
            }
            case DataFormat.JSON_TWO -> {
                ObjectMapper mapper = new ObjectMapper();
                for (int i = 0; i < inputData.length; i++) {
                    JsonTwo example = new JsonTwo(inputData[i], expectedResults[i]);
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

    /**
     * Randomly shuffles the training examples in the Dataset.
     */
    @SuppressWarnings("unused")
    public void shuffle(){
        Random rand = new Random();
        int indexOne, indexTwo;
        double[] tempArray;
        for (int i = 0; i < inputData.length; i++) {
            indexOne = rand.nextInt(inputData.length);
            indexTwo = rand.nextInt(inputData.length );

            tempArray = inputData[indexOne];
            inputData[indexOne] = inputData[indexTwo];
            inputData[indexTwo] = tempArray;

            tempArray = expectedResults[indexOne];
            expectedResults[indexOne] = expectedResults[indexTwo];
            expectedResults[indexTwo] = tempArray;
        }
    }

    /**
     * Merges list of datasets into one.
     * @param datasets list of datasets to be merged
     * @return one dataset containing all the examples from the datasets in the list
     */
    @SuppressWarnings("unused")
    public static Dataset mergeDatasets(List<Dataset> datasets){
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();
        for (Dataset dataset : datasets) {
            inputs.addAll(Arrays.asList(dataset.getInputData()));
            outputs.addAll(Arrays.asList(dataset.getExpectedResults()));
        }
        double[][] inputData = inputs.toArray(new double[inputs.size()][]);
        double[][] outputData = outputs.toArray(new double[outputs.size()][]);

        return new Dataset(inputData, outputData);
    }

    @SuppressWarnings("unused")
    public double[][] getInputData() {
        return inputData;
    }

    @SuppressWarnings("unused")
    public void setInputData(double[][] inputData) {
        this.inputData = inputData;
    }

    @SuppressWarnings("unused")
    public double[][] getExpectedResults() {
        return expectedResults;
    }

    @SuppressWarnings("unused")
    public void setExpectedResults(double[][] expectedResults) {
        this.expectedResults = expectedResults;
    }
}