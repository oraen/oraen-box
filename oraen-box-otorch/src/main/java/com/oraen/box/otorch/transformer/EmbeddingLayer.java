package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import lombok.Getter;

@Getter
public class EmbeddingLayer implements Layer<int[], double[]>, Learnable {

    private final int vocabSize;
    private final int embedDim;
    private final double[][] embeddingTable;

    public EmbeddingLayer(int vocabSize, int embedDim) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.embeddingTable = new double[vocabSize][embedDim];
    }

    // forward: token ids -> vectors
    @Override
    public double[][] forwardBatch(int[][] data) {
        return new double[0][];
    }

    // backward: grad vectors -> accumulate gradEmbedding
    @Override
    public int[][] backwardBatch(double[][] gradOutputBatch) {
        return new int[0][];
    }

    @Override
    public void updateParameters() {

    }

}