package com.codingbee.snn4j.interfaces.layers;

import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.RandomWeightGenerator;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.MethodCallingException;
import com.codingbee.snn4j.settings.TrainingSettings;

public interface Layer {
    float[][] process(float[][] i);

    void init(String path) throws FileManagingException;

    void init(RandomWeightGenerator randomGen);

    void save(String path) throws FileManagingException;

    void train(Dataset data, Model model) throws MethodCallingException;

    TrainingSettings getTrainingSettings();
    void setTrainingSettings(TrainingSettings settings);
}