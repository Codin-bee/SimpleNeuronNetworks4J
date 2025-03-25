package com.codingbee.snn4j.interface_implementations.models;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.model.Layer;
import com.codingbee.snn4j.interfaces.model.Model;
import com.codingbee.snn4j.interfaces.model.RandomWeightGenerator;
import com.codingbee.snn4j.settings.TrainingSettings;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LayeredModel implements Model {
    private List<Layer> layers = new ArrayList<>();
    private TrainingSettings trainingSettings = new TrainingSettings();
    int adamTime;

    public LayeredModel(){};

    @Override
    public float[][] process(float[][] input) {
        float[][] prevOutput = input;
        float[][] output = null;
        for (Layer layer : layers) {
            output = layer.process(prevOutput);
            prevOutput = output;
        }
        return output;
    }

    @Override
    public void init(RandomWeightGenerator randomGen) {
        for (Layer l : layers){
            l.init(randomGen);
        }
    }

    @Override
    public void init(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
            mapper.readerForUpdating(this).readValue(new File(path), LayeredModel.class);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred while trying to init the " +
                    "LayeredModel from file" + path + ": " + e.getLocalizedMessage());
        }
        for (Layer l : layers){
            l.setFullModel(this);
        }
    }

    @Override
    public void save(String path) throws FileManagingException{
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.writeValue(new File(path), this);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred trying to save the values of " +
                    "the LayeredModel: " + path + ": " + e.getLocalizedMessage());
        }
    }

    @Override
    public float calculateAverageCost(Dataset data) {
        float[][][] outputs = new float[data.getInputData().length][][];
        for (int i = 0; i < data.getInputData().length; i++) {
            outputs[i] = process(data.getInputData()[i]);
        }
        return Maths.calculateAverageMSE(outputs, data.getExpectedResults());
    }

    /**
     * Calculates correct percentage of the network on given data, assuming it processes classification tasks
     * and each output vector is evaluated individually
     * @param data the data to test the model on
     * @return the correctness of the model as a percentage
     */
    @Override
    public float calculateCorrectPercentage(Dataset data) {
        float[][][] outputs = new float[data.getInputData().length][][];
        for (int i = 0; i < data.getInputData().length; i++) {
            outputs[i] = process(data.getInputData()[i]);
        }
        return calculateCorrectPercentage(outputs, data.getExpectedResults());
    }

    @Override
    public void train(Dataset data, int epochs, boolean printDebug) {
        System.out.println("---Training---");
        for (Layer l : layers) {
            l.initAdamValues();
        }
        adamTime = 1;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            data.shuffle();
            List<Dataset> batches = data.splitIntoBatches();
            for (Dataset batch : batches) {
                for (Layer l : layers) {
                    l.prepareForwardPass(batch.getInputData().length);
                }
                float[][][] prevGradients = calculateInitialGradient(batch);
                for (int j = layers.size() - 1; j >= 0; j--) {
                    prevGradients = layers.get(j).backPropagateAndUpdate(prevGradients);
                }
                adamTime++;
            }
            if (printDebug){
                System.out.println("The cost after " + epoch + ". epoch is : " + calculateAverageCost(data));
            }
        }
    }

    @Override
    public TrainingSettings getTrainingSettings() {
        return trainingSettings;
    }

    @Override
    public void setTrainingSettings(TrainingSettings settings) {
        trainingSettings = settings;
        for (Layer l : layers){
            l.setTrainingSettings(settings);
        }
    }

    @Override
    public int getAdamTime() {
        return adamTime;
    }

    //region Private Methods
    private float calculateCorrectPercentage(float[][][] outputs, float[][][] expectedOutputs){
        float correct = 0;
        for (int i = 0; i < outputs.length; i++) {
            for (int j = 0; j < outputs[i].length; j++) {
                if (Maths.getIndexWithHighestVal(outputs[i][j]) ==
                        Maths.getIndexWithHighestVal(expectedOutputs[i][j])){
                    correct++;
                }
            }
        }
        return 100 * (correct / (outputs.length * outputs[0].length));
    }

    private float[][][] calculateInitialGradient(Dataset data){
        float[][][] gradients = new float[data.getInputData().length][][];

        for (int example = 0; example < data.getInputData().length; example++) {
            float[][] inputs = data.getInputData()[example];
            float[][] targets = data.getExpectedResults()[example];

            float[][] predictions = forwardPass(inputs);

            // MSE Loss Function
            float[][] exampleGradient = new float[predictions.length][predictions[0].length];
            for (int i = 0; i < predictions.length; i++) {
                for (int j = 0; j < predictions[i].length; j++) {
                    exampleGradient[i][j] = (float) Math.pow((predictions[i][j] - targets[i][j]), 2);
                }
            }
            gradients[example] = exampleGradient;
        }
        return gradients;
    }

    private float[][] forwardPass(float[][] input) {
        float[][] prevOutput = input;
        float[][] output = null;
        for (int i = 0; i < layers.size(); i++) {
            output = layers.get(i).forwardPass(prevOutput, i);
            prevOutput = output;
        }
        return output;
    }
    //endregion

    //region Getters and Setters for Jackson
    @JsonIgnore
    public int getNumberOfLayers(){
        return layers.size();
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    public void setAdamTime(int adamTime) {
        this.adamTime = adamTime;
    }

    public void addLayer(Layer layer) {
        layer.setFullModel(this);
        layers.add(layer);
    }

    public void addLayer(Layer layer, int index) {
        layer.setFullModel(this);
        layers.add(index, layer);
    }

    public void removeLayer(int index){
        layers.remove(index);
    }

    public void removeLayer(Layer layer){
        layers.remove(layer);
    }
    //endregion
}
