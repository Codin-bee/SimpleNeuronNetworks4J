package com.codingbee.snn4j.interfaces.model;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.settings.TrainingSettings;

public interface Layer {
    /**
     * Processes given input array and returns a new output array.
     * @param input 2D input array
     * @return 2D output array
     */
    float[][] process(float[][] input);

    /**
     * Initializes the parameters of the layer from file.
     * @param path path to the file, which stores the parameters
     * @throws FileManagingException if any Exception occurs, while trying to initialize the parameters
     */
    void init(String path) throws FileManagingException;

    /**
     * Initializes the parameters of the layer using random weight generator
     * @param randomGen the generator to use for the random parameter generating
     */
    void init(RandomWeightGenerator randomGen);

    /**
     * Saves the parameters of the layer into given file.
     * @param path path to the file to save
     * @throws FileManagingException if any Exception occurs while trying to save the parameters
     */
    void save(String path) throws FileManagingException;

    void initAdamValues();

    /**
     * Fine-tunes the layer parameters on given dataset to minimize the cost function of the model it is part of.
     * @param data the dataset to train on
     * @param model the model the layer is part of
     * @throws IncorrectDataException if any illegal argument is passed
     */
    void train(Dataset data, Model model) throws IncorrectDataException;

    TrainingSettings getTrainingSettings();
    void setTrainingSettings(TrainingSettings settings);

    Model getFullModel();
    void setFullModel(Model fullModel);
}