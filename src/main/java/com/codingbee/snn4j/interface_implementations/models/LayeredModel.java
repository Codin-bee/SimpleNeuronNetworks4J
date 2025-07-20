package com.codingbee.snn4j.interface_implementations.models;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interface_implementations.cost_functions.MeanSquaredError;
import com.codingbee.snn4j.interfaces.utils.CostFunction;
import com.codingbee.snn4j.interfaces.architecture.Layer;
import com.codingbee.snn4j.interfaces.architecture.Model;
import com.codingbee.snn4j.interfaces.utils.RandomWeightGenerator;
import com.codingbee.snn4j.interfaces.utils.VectorActivationFunction;
import com.codingbee.snn4j.settings.TrainingSettings;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;

public class LayeredModel implements Model {
    private Layer[] layers = new Layer[0];
    private TrainingSettings trainingSettings = new TrainingSettings();
    private CostFunction costFunction = new MeanSquaredError();
    private VectorActivationFunction outputActivationFunction = null;

    public LayeredModel(){};

    public LayeredModel(Layer[] layers){
        //TODO: Exceptions, rework for better time
        for (Layer layer : layers) {
            addLayer(layer);
        }
    }

    public LayeredModel(List<Layer> layers){
        //TODO: Exceptions
        for (Layer layer : layers){
            addLayer(layer);
        }
    }

    @Override
    public float[][] process(float[][] input) {
        float[][] prevOutput = input;
        float[][] output = null;
        for (Layer layer : layers) {
            output = layer.process(prevOutput);
            prevOutput = output;
        }
        if (outputActivationFunction != null) {
            for (int i = 0; i < Objects.requireNonNull(output).length; i++) {
                output[i] = outputActivationFunction.activate(output[i]);
            }
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
        return costFunction.calculateAverage(outputs, data.getExpectedResults());
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
    public void train(Dataset data, int epochs, String savePath, int saveInterval, boolean printDebug) throws FileManagingException {
        System.out.println("---Training---");
        for (Layer l : layers) {
            l.initAdamValues();
            l.setTrainingSettings(trainingSettings);
        }
        int adamTime = 1;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            data.shuffle();
            List<Dataset> batches = data.splitIntoBatches();
            for (Dataset batch : batches) {
                for (Layer l : layers) {
                    l.prepareForwardPass(batch.getInputData().length);
                }
                float[][][] prevGradients = calculateInitialGradient(batch);
                for (int j = layers.length - 1; j >= 0; j--) {
                    prevGradients = layers[j].backPropagateAndUpdate(prevGradients, adamTime);
                }
                adamTime++;
            }
            if (printDebug){
                System.out.println("The cost after " + epoch + ". epoch is : " + calculateAverageCost(data)
                 + ", the correct percentage is: " + calculateCorrectPercentage(data));
            }
            if ((epoch % saveInterval) == 0){
                save(savePath);
            }
        }
        save(savePath);
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


    //region Private Methods
    private float calculateCorrectPercentage(float[][][] outputs, float[][][] expectedOutputs){
        float correct = 0;
        for (int i = 0; i < outputs.length; i++) {
            for (int j = 0; j < outputs[i].length; j++) {
                if (Maths.getIndexOfLargestElement(outputs[i][j]) ==
                        Maths.getIndexOfLargestElement(expectedOutputs[i][j])){
                    correct++;
                }
            }
        }
        return 100 * (correct / (outputs.length * outputs[0].length));
    }

    private float[][][] calculateInitialGradient(Dataset data){
        float[][][] gradients = new float[data.getInputData().length][][];
        float[][][] outputs = new float[data.getInputData().length][][];

        for (int example = 0; example < data.getInputData().length; example++) {
            float[][] inputs = data.getInputData()[example];
            float[][] targets = data.getExpectedResults()[example];

            float[][] predictions = forwardPass(inputs, example);
            if (outputActivationFunction != null){
                outputs[example] = predictions;
            }

            float[][] exampleGradient = new float[predictions.length][predictions[0].length];
            for (int i = 0; i < predictions.length; i++) {
                for (int j = 0; j < predictions[i].length; j++) {
                    exampleGradient[i][j] = costFunction.calculateDerivative(predictions[i][j], targets[i][j]);
                }
            }
            gradients[example] = exampleGradient;
        }
        if (outputActivationFunction != null) {
            for (int i = 0; i < gradients.length; i++) {
                for (int j = 0; j < gradients[i].length; j++) {
                    gradients[i][j] = outputActivationFunction.derivative(outputs[i][j], gradients[i][j]);
                }
            }
        }
        return gradients;
    }

    private float[][] forwardPass(float[][] input, int example) {
        float[][] prevOutput = input;
        float[][] output = null;
        for (Layer layer : layers) {
            output = layer.forwardPass(prevOutput, example);
            prevOutput = output;
        }
        if (outputActivationFunction != null) {
            for (int i = 0; i < Objects.requireNonNull(output).length; i++) {
                output[i] = outputActivationFunction.activate(output[i]);
            }
        }
        return output;
    }

    private boolean layersMatchDimensions(Layer first, Layer second){
        return first.getOutputD() == second.getInputD();
    }
    //endregion

    //region Getters and Setters for Jackson:
    //Jackson has trouble deserializing the value without this annotation
    @JsonIgnore
    public int getNumberOfLayers(){
        if (layers == null){
            return 0;
        }
        return layers.length;
    }

    public Layer[] getLayers() {
        return layers;
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }

    public void addLayer(Layer layer) {
        if (layers.length != 0){
            if (layersMatchDimensions(layers[layers.length - 1], layer)){
                throw new IncorrectDataException("The added layer does not match the dimensions of the previous layer");
            }
        }

        Layer[] newLayers = new Layer[layers.length + 1];
        System.arraycopy(layers, 0, newLayers, 0, layers.length);
        newLayers[newLayers.length - 1] = layer;

        layers = newLayers;
    }

    public void removeLayer(int index){
        if (layers.length == 0){
            throw new IncorrectDataException("There are no layers in the model");
        }

        Layer[] newLayers = new Layer[layers.length - 1];
        int j = 0;
        for (int i = 0; i < layers.length; i++) {
            if (i == index){
                continue;
            }
            newLayers[j] = layers[i];
            j++;
        }

        layers = newLayers;
    }

    public CostFunction getCostFunction() {
        return costFunction;
    }

    public void setCostFunction(CostFunction costFunction) {
        this.costFunction = costFunction;
    }

    public VectorActivationFunction getOutputActivationFunction() {
        return outputActivationFunction;
    }

    public void setOutputActivationFunction(VectorActivationFunction outputActivationFunction) {
        this.outputActivationFunction = outputActivationFunction;
    }

    //endregion
}