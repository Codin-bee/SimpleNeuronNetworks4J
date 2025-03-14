package com.codingbee.snn4j.interface_implementations.layers;

import com.codingbee.snn4j.algorithms.Maths;
import com.codingbee.snn4j.exceptions.FileManagingException;
import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.helping_objects.Dataset;
import com.codingbee.snn4j.interfaces.ActivationFunction;
import com.codingbee.snn4j.interfaces.model.Layer;
import com.codingbee.snn4j.interfaces.model.Model;
import com.codingbee.snn4j.interfaces.model.RandomWeightGenerator;
import com.codingbee.snn4j.settings.TrainingSettings;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class LSTMLayer implements Layer {
    private TrainingSettings trainingSettings = new TrainingSettings();
    private Model fullModel;

    private ActivationFunction cellStateAF;
    private ActivationFunction forgetGateAF;
    private ActivationFunction inputGateAF;
    private ActivationFunction outputGateAF;
    private ActivationFunction cellCandidateAF;

    private int numberOfCells;
    private int d_input;
    private int d_output;

    private float[][][] forgetGateW;
    private float[][][] inputGateW;
    private float[][][] outputGateW;
    private float[][][] cellCandidateW;

    private float[][] forgetGateB;
    private float[][] inputGateB;
    private float[][] outputGateB;
    private float[][] cellCandidateB;

    private float[][] prevHiddenStates;
    private float[][] prevCellStates;


    @Override
    public float[][] process(float[][] input) {
        for (int i = 0; i < numberOfCells; i++) {
            Arrays.fill(prevHiddenStates[i], 0);
            Arrays.fill(prevCellStates[i], 0);
        }

        float[][] outputMatrix = new float[input.length][];
        for (int i = 0; i < input.length; i++){
            //Processing input vectors sequentially
            float[] combinedOutput = new float[input[0].length];
            for (int j = 0; j < numberOfCells; j++) {
                float[] cellOutput = calculateCellOutput(j, input[i]);
                //Adding output vectors to average them
                //Maybe could be concatenated
                for (int k = 0; k < input[0].length; k++) {
                    combinedOutput[k] += cellOutput[k];
                }
            }
            for (int j = 0; j < combinedOutput.length; j++) {
                combinedOutput[j] /= numberOfCells;
            }
            outputMatrix[i] = combinedOutput;
        }
        return outputMatrix;
    }

    @Override
    public void init(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
            mapper.readerForUpdating(this).readValue(new File(path), LSTMLayer.class);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred while trying to init the " +
                    "LSTMLayer from file: " + path + ": " + e.getLocalizedMessage());
        }
    }

    @Override
    public void init(RandomWeightGenerator randomGen) {
        for (int i = 0; i < numberOfCells; i++) {
            for (int j = 0; j < d_output; j++) {
                forgetGateB[i][j] = randomGen.getHiddenLayerBias();
                inputGateB[i][j] = randomGen.getHiddenLayerBias();
                outputGateB[i][j] = randomGen.getHiddenLayerBias();
                cellCandidateB[i][j] = randomGen.getHiddenLayerBias();
            }

            for (int j = 0; j < d_output; j++) {
                for (int k = 0; k < d_output + d_input; k++) {
                    forgetGateW[i][j][k] = randomGen.getWeight(d_input, d_output);
                    inputGateW[i][j][k] = randomGen.getWeight(d_input, d_output);
                    outputGateW[i][j][k] = randomGen.getWeight(d_input, d_output);
                    cellCandidateW[i][j][k] = randomGen.getWeight(d_input, d_output);
                }
            }
        }
    }

    @Override
    public void save(String path) throws FileManagingException {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.writeValue(new File(path), this);
        } catch (IOException e) {
            throw new FileManagingException("An Exception occurred trying to save the values of " +
                    "the Dataset" + e.getLocalizedMessage());
        }
    }

    @Override
    public void initAdamValues() {

    }

    @Override
    public void train(Dataset data, Model model) throws IncorrectDataException {

    }
    //region Private methods
    private float[] calculateCellOutput(int cellNo, float[] input){
        float[] processedVector = Maths.concatenateVectors(input, prevHiddenStates[cellNo]);

        float[] inputGate = calculateGate(inputGateW[cellNo], inputGateB[cellNo], inputGateAF, processedVector);
        float[] forgetGate = calculateGate(forgetGateW[cellNo], forgetGateB[cellNo], forgetGateAF, processedVector);
        float[] outputGate = calculateGate(outputGateW[cellNo], outputGateB[cellNo], outputGateAF, processedVector);
        float[] cellCandidate = calculateGate(cellCandidateW[cellNo], cellCandidateB[cellNo], cellCandidateAF, processedVector);

        float[] cellState = Maths.addVectors(Maths.multiplyVectors(forgetGate, prevCellStates[cellNo]), Maths.multiplyVectors(inputGate, cellCandidate));
        for (int i = 0; i < cellState.length; i++) {
            cellState[i] = cellStateAF.activate(cellState[i]);
        }

        float[] activatedCellState = new float[cellState.length];
        for (int i = 0; i < activatedCellState.length; i++) {
            activatedCellState[i] = cellStateAF.activate(cellState[i]);
        }
        float[] hiddenState = Maths.multiplyVectors(outputGate, activatedCellState);

        //Cleanup
        prevHiddenStates[cellNo] = hiddenState;
        prevCellStates[cellNo] = cellState;

        return hiddenState;
    }

    private float[] calculateGate(float[][] weightMatrix, float[] biasVector, ActivationFunction activation, float[] vector){
        float[] gate = Maths.multiplyWbyV(weightMatrix, vector);
        for (int i = 0; i < gate.length; i++) {
            activation.activate(gate[i] += biasVector[i]);
        }
        return gate;
    }
    //endregion

    //region Getters and Setters
    public ActivationFunction getCellStateAF() {
        return cellStateAF;
    }

    public void setCellStateAF(ActivationFunction cellStateAF) {
        this.cellStateAF = cellStateAF;
    }

    public ActivationFunction getForgetGateAF() {
        return forgetGateAF;
    }

    public void setForgetGateAF(ActivationFunction forgetGateAF) {
        this.forgetGateAF = forgetGateAF;
    }

    public ActivationFunction getOutputGateAF() {
        return outputGateAF;
    }

    public void setOutputGateAF(ActivationFunction outputGateAF) {
        this.outputGateAF = outputGateAF;
    }

    public ActivationFunction getCellCandidateAF() {
        return cellCandidateAF;
    }

    public void setCellCandidateAF(ActivationFunction cellCandidateAF) {
        this.cellCandidateAF = cellCandidateAF;
    }

    public ActivationFunction getInputGateAF() {
        return inputGateAF;
    }

    public void setInputGateAF(ActivationFunction inputGateAF) {
        this.inputGateAF = inputGateAF;
    }

    //endregion

    //region Basic Interface Getters and Setters
    @Override
    public TrainingSettings getTrainingSettings() {
        return trainingSettings;
    }

    @Override
    public void setTrainingSettings(TrainingSettings settings) {
        trainingSettings = settings;
    }

    @Override
    public Model getFullModel() {
        return fullModel;
    }

    @Override
    public void setFullModel(Model fullModel) {
        this.fullModel = fullModel;
    }
    //endregion

    //region Weight, Bias and Cell State Getters and Setters for Jackson

    public float[][][] getForgetGateW() {
        return forgetGateW;
    }

    public void setForgetGateW(float[][][] forgetGateW) {
        this.forgetGateW = forgetGateW;
    }

    public float[][][] getInputGateW() {
        return inputGateW;
    }

    public void setInputGateW(float[][][] inputGateW) {
        this.inputGateW = inputGateW;
    }

    public float[][][] getOutputGateW() {
        return outputGateW;
    }

    public void setOutputGateW(float[][][] outputGateW) {
        this.outputGateW = outputGateW;
    }

    public float[][][] getCellCandidateW() {
        return cellCandidateW;
    }

    public void setCellCandidateW(float[][][] cellCandidateW) {
        this.cellCandidateW = cellCandidateW;
    }

    public float[][] getForgetGateB() {
        return forgetGateB;
    }

    public void setForgetGateB(float[][] forgetGateB) {
        this.forgetGateB = forgetGateB;
    }

    public float[][] getInputGateB() {
        return inputGateB;
    }

    public void setInputGateB(float[][] inputGateB) {
        this.inputGateB = inputGateB;
    }

    public float[][] getOutputGateB() {
        return outputGateB;
    }

    public void setOutputGateB(float[][] outputGateB) {
        this.outputGateB = outputGateB;
    }

    public float[][] getCellCandidateB() {
        return cellCandidateB;
    }

    public void setCellCandidateB(float[][] cellCandidateB) {
        this.cellCandidateB = cellCandidateB;
    }

    public float[][] getPrevHiddenStates() {
        return prevHiddenStates;
    }

    public void setPrevHiddenStates(float[][] prevHiddenStates) {
        this.prevHiddenStates = prevHiddenStates;
    }

    public float[][] getPrevCellStates() {
        return prevCellStates;
    }

    public void setPrevCellStates(float[][] prevCellStates) {
        this.prevCellStates = prevCellStates;
    }


    //endregion
}
