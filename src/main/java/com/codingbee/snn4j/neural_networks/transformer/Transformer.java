package com.codingbee.snn4j.neural_networks.transformer;

@SuppressWarnings("unused")
public class Transformer {
    int contextSize;
    int layers;
    int layerHeads;
    int d_model;
    int d_ffn;
    int d_attention;
    int attentionScalingFactor;

    public Transformer(int contextSize, int layers, int layerHeads, int d_model, int d_ffn){
        this.contextSize = contextSize;
        this.layers = layers;
        this.layerHeads = layerHeads;
        this.d_model = d_model;
        this.d_ffn = d_ffn;
        d_attention = d_model / layerHeads;
        attentionScalingFactor = (int) Math.sqrt(d_attention);
    }
}
