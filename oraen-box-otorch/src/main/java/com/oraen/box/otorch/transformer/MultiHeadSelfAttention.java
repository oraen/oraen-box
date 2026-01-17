package com.oraen.box.otorch.transformer;

import lombok.Getter;

@Getter
public class MultiHeadSelfAttention implements AttentionLayer {

    private final int embedDim;
    private final int numHeads;

    public MultiHeadSelfAttention(int embedDim, int numHeads) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
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

    // Wq, Wk, Wv, Wo
    // forward: [B, T, D] -> [B, T, D]
}