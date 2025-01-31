package com.codingbee.snn4j.interfaces.layers;

import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;
import com.codingbee.snn4j.settings.DebuggingSettings;
import com.codingbee.snn4j.settings.TrainingSettings;

public interface Model {
    void addLayer(Layer layer);
    void addLayer(Layer layer, int index);

    float[][] process(float[][] i);

    void init(RandomWeightGenerator randomGen);
    void init();
    void init(String path);

    void save();
    void save(String path);

    float calculateAverageCost(Dataset data);
    float calculateAverageCost(float[][][] input, float[][][] expectedOutput);

    float calculateCorrectPercentage(Dataset data);
    float calculateCorrectPercentage(float[][][] input, float[][][] expectedOutput);

    void train(Dataset data, int epochs, boolean printDebug);

    void setDebuggingSettings(DebuggingSettings setting);
    DebuggingSettings getDebuggingSettings();

    TrainingSettings getTrainingSettings();
    void setTrainingSettings(TrainingSettings settings);
}
