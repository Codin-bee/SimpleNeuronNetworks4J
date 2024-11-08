package com.codingbee.snn4j.neural_networks.transformer_encoder;

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

    //Embedding matrix
    double[][] embeddingMatrix = new double[dictionarySize][d_model];
    //Attention matrices
    double[][][][] keyMatrices = new double[layers][layerHeads][d_model][d_attention];
    double[][][][] quarryMatrices = new double[layers][layerHeads][d_model][d_attention];
    double[][][][] valueMatrices = new double[layers][layerHeads][d_model][d_model];
    //FFN weights and biases
    double[][][][] ffnWeight = new double[layers][][][];
    double[][][] ffnBiases = new double[layers][][];
    //Layer normalization parameters
    double[][] betas = new double[layers][d_model];
    double[][] gammas = new double[layers][d_model];

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

    }

    private void applyNormalization(int layer, int type, double[][] values){

    }

    private double[][] calculateOutput(double[][] embeddings){
        return null;
    }

    private double[][] addMatrices(double[][] matrixA, double[][] matrixB){
        double[][] matrixC = new double[matrixA.length][];
        return matrixC;
    }
    //endregion
}
