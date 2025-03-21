package com.codingbee.snn4j.interface_implementations.models;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.model.Layer;
import com.codingbee.snn4j.interfaces.model.Model;
import com.codingbee.snn4j.interfaces.model.RandomWeightGenerator;
import com.codingbee.snn4j.settings.TrainingSettings;

import java.io.File;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class LayeredModel implements Model {
    private final List<Layer> layers = new ArrayList<>();
    private TrainingSettings trainingSettings = new TrainingSettings();
    int adamTime;

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void addLayer(Layer layer, int index) {
        layers.add(index, layer);
    }

    public void removeLayer(int index){
        layers.remove(index);
    }

    public void removeLayer(Layer layer){
        layers.remove(layer);
    }

    public int getNumberOfLayers(){
        return layers.size();
    }

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
        File directory = new File(path);
        if (!directory.isDirectory()) throw new FileManagingException("There is no directory at: " + path);
        for (int i = 0; i < layers.size(); i++){
            layers.get(i).init(path + i + ".json");
        }
    }

    @Override
    public void save(String path) throws FileManagingException{
        File directory = new File(path);
        if (!(directory.isDirectory() || directory.mkdirs())) throw new FileManagingException("Could not " +
                "create required directory: " + path);
        for (int i = 0; i < layers.size(); i++){
            layers.get(i).save(path + File.separator +  i + ".json");
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
        for (Layer l : layers) {
            l.initAdamValues();
        }
        adamTime = 1;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            data.shuffle();
            List<Dataset> batches = data.splitIntoBatches();
            for (Dataset batch : batches) {
                float[][][] prevGradients = calculateInitialGradient(batch);
                for (int j = layers.size() - 1; j >= 0; j--) {
                    //Replace NULL by the layerInput
                    prevGradients = layers.get(j).backPropagateAndUpdate(prevGradients, null);
                }
                adamTime++;
            }
            if (printDebug){
                System.out.println(LocalDateTime.now());
                System.out.println("The cost after " + epoch + ". epoch is : " + calculateAverageCost(data));
                System.out.println(LocalDateTime.now());
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

            float[][] predictions = process(inputs);

            // Mean squared error loss function
            float[][] exampleGradient = new float[predictions.length][predictions[0].length];
            for (int i = 0; i < predictions.length; i++) {
                for (int j = 0; j < predictions[i].length; j++) {
                    exampleGradient[i][j] = 2 * (predictions[i][j] - targets[i][j]); // MSE gradient
                }
            }
            gradients[example] = exampleGradient;
        }
        return gradients;
    }
    //endregion
}
