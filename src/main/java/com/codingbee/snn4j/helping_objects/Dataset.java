package com.codingbee.snn4j.helping_objects;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.*;

@SuppressWarnings("unused")
public class Dataset {
    private float[][][] inputData;
    private float[][][] expectedResults;

    /**
     * Object which holds data used for training or testing any model
     * @param inputData the values used as input, array of 2D arrays, where
     *                  each 2D array represents one example
     * @param expectedResults the values expected in the output of the model,
     *                        formated the same way as inputs
     */
    @SuppressWarnings("unused")
    public Dataset(float[][][] inputData, float[][][] expectedResults) {
        this.inputData = inputData;
        this.expectedResults = expectedResults;
    }

    /**
     * Creates new instance of Dataset and loads the values from json file into it
     * @param path path to the json file
     * @throws IOException if any Exception occurs while working with files
     */
    public Dataset(String path) throws IOException {
        this.loadFromJson(path);
    }

    /**
     * Empty constructor for Dataset used mainly for loading values from files
     */
    public Dataset(){
    }

    /**
     * Saves the dataset values as a json file
     * @param path path to the json file to be created
     * @throws FileManagingException if any Exception occurs while trying to save the values
     * of the Dataset
     */
    public void saveAsJson(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.writeValue(new File(path), this);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred trying to save the values of " +
                    "the Dataset" + e.getLocalizedMessage());
        }
    }

    /**
     * Loads the Dataset values from the json file at given path
     * @param path path to the json file
     * @throws FileManagingException if any Exception occurs while loading the values from file
     */
    public void loadFromJson(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            Dataset data = mapper.readValue(new File(path), Dataset.class);
            this.expectedResults = data.expectedResults;
            this.inputData = data.inputData;
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred while trying to load the " +
                    "Dataset from file" + e.getLocalizedMessage());
        }
    }

    /**
     * Randomly shuffles the training examples in the Dataset.
     */
    public void shuffle(){
        Random rand = new Random();
        int indexOne, indexTwo;
        float[][] temp;
        for (int i = 0; i < inputData.length; i++) {
            indexOne = rand.nextInt(inputData.length);
            indexTwo = rand.nextInt(inputData.length );

            temp = inputData[indexOne];
            inputData[indexOne] = inputData[indexTwo];
            inputData[indexTwo] = temp;

            temp = expectedResults[indexOne];
            expectedResults[indexOne] = expectedResults[indexTwo];
            expectedResults[indexTwo] = temp;
        }
    }

    /**
     * Merges list of datasets into one.
     * @param datasets list of datasets to be merged
     * @return one dataset containing all the examples from the datasets in the list
     */
    public static Dataset mergeDatasets(List<Dataset> datasets){
        List<float[][]> inputs = new ArrayList<>();
        List<float[][]> outputs = new ArrayList<>();
        for (Dataset dataset : datasets) {
            inputs.addAll(Arrays.asList(dataset.getInputData()));
            outputs.addAll(Arrays.asList(dataset.getExpectedResults()));
        }
        float[][][] inputData = inputs.toArray(new float[inputs.size()][][]);
        float[][][] outputData = outputs.toArray(new float[outputs.size()][][]);

        return new Dataset(inputData, outputData);
    }

    //region Getters and setters
    public float[][][] getInputData() {
        return inputData;
    }

    public void setInputData(float[][][] inputData) {
        this.inputData = inputData;
    }

    public float[][][] getExpectedResults() {
        return expectedResults;
    }

    public void setExpectedResults(float[][][] expectedResults) {
        this.expectedResults = expectedResults;
    }
    //endregion
}