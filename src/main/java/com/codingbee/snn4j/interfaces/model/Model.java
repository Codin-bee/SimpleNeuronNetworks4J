package com.codingbee.snn4j.interfaces.model;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.settings.TrainingSettings;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.MINIMAL_CLASS, property = "@class")
public interface Model {
    float[][] process(float[][] i);

    void init(RandomWeightGenerator randomGen);
    void init(String path) throws FileManagingException;

    void save(String path) throws FileManagingException;

    float calculateAverageCost(Dataset data);

    float calculateCorrectPercentage(Dataset data);

    void train(Dataset data, int epochs, boolean printDebug);

    TrainingSettings getTrainingSettings();
    void setTrainingSettings(TrainingSettings settings);

    int getAdamTime();
}
