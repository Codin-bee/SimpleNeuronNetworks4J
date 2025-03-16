package com.codingbee.snn4j.interfaces.model;

import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.settings.TrainingSettings;

public interface Model {
    void addLayer(Layer layer);
    void addLayer(Layer layer, int index);

    float[][] process(float[][] i);

    void init(RandomWeightGenerator randomGen);
    void init(String path);

    void save(String path);

    float calculateAverageCost(Dataset data);

    float calculateCorrectPercentage(Dataset data);

    void train(Dataset data, int epochs, boolean printDebug);

    TrainingSettings getTrainingSettings();
    void setTrainingSettings(TrainingSettings settings);

    int getTrainingTime();
}
