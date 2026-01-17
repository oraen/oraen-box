package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import lombok.Data;

@Data
public class TransformerEncoderBlock implements Layer<double[][], double[][]>, Learnable {

    private MultiHeadSelfAttention selfAttention;
    private LayerNorm ln1;

    private FeedForwardLayer ffn;
    private LayerNorm ln2;

    public TransformerEncoderBlock(MultiHeadSelfAttention selfAttention, LayerNorm ln1, FeedForwardLayer ffn, LayerNorm ln2) {
        this.selfAttention = selfAttention;
        this.ln1 = ln1;
        this.ffn = ffn;
        this.ln2 = ln2;
    }

    @Override
    public double[][][] forwardBatch(double[][][] data) {
        return new double[0][][];
    }

    @Override
    public double[][][] backwardBatch(double[][][] gradOutputBatch) {
        return new double[0][][];
    }

    @Override
    public void updateParameters() {

    }
}