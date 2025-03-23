package com.codingbee.snn4j.interfaces.model;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.settings.TrainingSettings;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.MINIMAL_CLASS, property = "@class")
public interface Layer {
    /**
     * Processes given input array and returns a new output array
     * @param input 3D input array (batchSize x D1 x D2)
     * @return 3D output array (batchSize x D1 x D2)
     */
    float[][] process(float[][] input);

    /**
     * Processes given input array and returns a new output array. Keeps all the values stored inside
     * @param input 3D input array (batchSize x D1 x D2)
     * @return 3D output array (batchSize x D1 x D2)
     */
    float[][] forwardPass(float[][] input, int index);

    /**
     * Prepares the layer for forward pass, allocates all arrays
     * @param numberOfSamples number of sample in the next forward pass
     */
    void prepareForwardPass(int numberOfSamples);

    /**
     * Performs backpropagation to adjust weights based on the error gradient
     * @param outputErrors 3D error array from the next layer (batchSize x D1 x D2)
     * @return 3D error array to propagate to the previous layer (batchSize x D1 x D2)
     */
    float[][][] backPropagateAndUpdate(float[][][] outputErrors);

    /**
     * Initializes the parameters of the layer from given json file
     * @param path path to the file, which stores the parameters
     * @throws FileManagingException if any Exception occurs while trying to initialize the parameters
     */
    void init(String path) throws FileManagingException;

    /**
     * Initializes the parameters of the layer using a random weight generator.
     * @param randomGen the generator to use for the random parameter generating
     */
    void init(RandomWeightGenerator randomGen);

    /**
     * Saves the parameters of the layer into a given json file
     * @param path path to the file to save
     * @throws FileManagingException if any Exception occurs while trying to save the parameters
     */
    void save(String path) throws FileManagingException;

    /**
     * Initializes values for Adam optimizer
     */
    void initAdamValues();

    /**
     * Gets the training settings for this layer
     * @return Training settings
     */
    TrainingSettings getTrainingSettings();

    /**
     * Sets the training settings for this layer
     * @param settings Training settings
     */
    void setTrainingSettings(TrainingSettings settings);

    /**
     * Gets the full model this layer is part of
     * @return Full model
     */
    Model getFullModel();

    /**
     * Sets the full model this layer is part of
     * @param fullModel Full model
     */
    void setFullModel(Model fullModel);
}
