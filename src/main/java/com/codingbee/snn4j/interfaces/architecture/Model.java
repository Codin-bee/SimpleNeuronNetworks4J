package com.codingbee.snn4j.interfaces.architecture;

import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.utils.RandomWeightGenerator;
import com.codingbee.snn4j.settings.TrainingSettings;

public interface Model {
    float[][] process(float[][] i);

    void init(RandomWeightGenerator randomGen);

    void init(String path) throws FileManagingException;
    void save(String path) throws FileManagingException;

    float calculateAverageCost(Dataset data);
    float calculateCorrectPercentage(Dataset data);

    void train(Dataset data, int epochs, String savePath, int saveInterval, boolean printDebug) throws FileManagingException;

    void validate();

    TrainingSettings getTrainingSettings();
    void setTrainingSettings(TrainingSettings settings);
}
