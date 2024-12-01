package com.codingbee.snn4j.neural_networks.transformer_encoder;

import com.codingbee.snn4j.exceptions.IncorrectDataException;
import com.codingbee.snn4j.exceptions.MethodCallingException;

import java.util.Arrays;

@SuppressWarnings("unused")
public class TransformerEncoder {
    //Dimensions, sizes, etc.
    int dictionarySize;
    int contextSize;
    int layers;
    int layerHeads;
    int d_model;
    int d_ffn;
    int d_attention;
    double attentionScalingFactor;
    boolean initialized;

    //Embedding matrix
    double[][] embeddingMatrix = new double[dictionarySize][d_model];
    //Attention matrices
    double[][][][] keyMatrices = new double[layers][layerHeads][d_model][d_attention];
    double[][][][] quarryMatrices = new double[layers][layerHeads][d_model][d_attention];
    double[][][][] valueMatrices = new double[layers][layerHeads][d_model][d_model];
    //FFN weights and biases
    double[][][][] ffnWeights = new double[layers][][][];
    double[][][] ffnBiases = new double[layers][][];
    //Layer normalization parameters
    double[][] betas = new double[layers][d_model];
    double[][] gammas = new double[layers][d_model];
    //Unembedding matrix
    double[][] unembeddingMatrix = new double[d_model][];

    public TransformerEncoder(int dictionarySize, int contextSize, int layers, int layerHeads, int d_model, int d_ffn) {
        this.dictionarySize = dictionarySize;
        this.contextSize = contextSize;
        this.layers = layers;
        this.layerHeads = layerHeads;
        this.d_model = d_model;
        this.d_ffn = d_ffn;

        d_attention = d_model / layerHeads;
        attentionScalingFactor = 1 / Math.sqrt(d_attention);
    }

    public double[][] processSequenceAsValues(int[] sequence) {
        if (!initialized){
            throw new MethodCallingException("The network can not process any values because it has not been initialized properly");
        }
        //Embedding
        double[][] embeddings = embed(sequence);
        applyPosEncoding(embeddings);
        double[][] subLayerInput;

        for (int i = 0; i < layers; i++) {
            //Attention
            subLayerInput = Arrays.copyOf(embeddings, embeddings.length);
            applyNormalization(i, 0, embeddings);
            applyAttention(i, embeddings);
            embeddings = addMatrices(subLayerInput, embeddings);

            //FFNs
            subLayerInput = Arrays.copyOf(embeddings, embeddings.length);
            applyNormalization(i, 1, embeddings);
            applyFFNs(i, embeddings);
            embeddings = addMatrices(subLayerInput, embeddings);
        }
        //Output
        return calculateOutput(embeddings);
    }


    //region Private Methods
    private double[][] embed(int[] indices) {
        if (embeddingMatrix == null){
            throw new MethodCallingException("The called MatrixEmbedding does not contain valid translation matrix");
        }
        double[][] values = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            values[i] = new double[embeddingMatrix[0].length];
            System.arraycopy(embeddingMatrix[indices[i]], 0, values[i], 0, values[i].length);
        }
        return values;
    }

    private void applyPosEncoding(double[][] values){

    }

    private void applyAttention(int layer, double[][] values){

    }

    private void applyFFNs(int layer, double[][] values){
        for (int i = 0; i < values.length; i++) {
            //apply for each vector
        }
    }

    private void applyNormalization(int layer, int type, double[][] values){

    }

    private double[][] calculateOutput(double[][] embeddings){
        return multiplyMatrices(embeddings, unembeddingMatrix);
    }

    private double[][] addMatrices(double[][] matrixA, double[][] matrixB){
        if (matrixA == null || matrixB == null){
            throw new IncorrectDataException("The passed matrices can not be null");
        }
        if (matrixA.length != matrixB.length){
            throw new IncorrectDataException("The matrices have to have same dimensions in order to add them," +
                    " but the number of rows in each is not the same");
        }
        if (matrixA[0].length != matrixB[0].length){
            throw new IncorrectDataException("The matrices have to have same dimensions in order to add them," +
                    " but the number of columns in each is not the same");
        }

        double[][] matrixC = new double[matrixA.length][matrixA[0].length];
        try {
            for (int i = 0; i < matrixA.length; i++) {
                for (int j = 0; j < matrixA[0].length; j++) {
                    matrixC[i][j] = matrixA[i][j] + matrixB[i][j];
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new IncorrectDataException("The passed matrices need to have" +
                    " same number of columns and rows on each index: " + e.getLocalizedMessage());
        }
        return  matrixC;
    }

    private double[][] multiplyMatrices(double[][] matrixA, double[][] matrixB){

        if (matrixA == null || matrixB == null){
            throw new IncorrectDataException("The passed matrices can not be null");
        }
        if (matrixA[0].length != matrixB.length){
            throw new IncorrectDataException("The number of columns in first matrix must be equal to " +
                    "the number of rows in second matrix in order to multiply them");
        }
        double[][] matrixC = new double[matrixA.length][matrixB[0].length];
        try {
            for (int i = 0; i < matrixA.length; i++) {
                for (int j = 0; j < matrixB[0].length; j++) {
                    for (int k = 0; k < matrixB.length; k++) {
                        matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                    }
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new IncorrectDataException("The passed matrices need to have" +
                    " same number of columns and rows on each index: " + e.getLocalizedMessage());
        }
        return matrixC;
    }
    //endregion
}
