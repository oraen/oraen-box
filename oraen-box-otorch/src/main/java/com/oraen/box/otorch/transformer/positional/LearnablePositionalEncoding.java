// LearnablePositionalEncoding.java
package com.oraen.box.otorch.transformer.positional;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.transformer.PositionalEncodingLayer;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

@Getter
public class LearnablePositionalEncoding implements PositionalEncodingLayer, Learnable {

    private final int maxSeqLen;
    private final int dim;
    private final double[][] posEmbedding;      // [maxSeqLen, dim]
    private final double[][] gradPosEmbedding;  // [maxSeqLen, dim]

    @Setter
    private GradOptimizer optimizer;

    public LearnablePositionalEncoding(int maxSeqLen, int dim, GradOptimizer optimizer, ParamInitializer initializer) {
        this.maxSeqLen = maxSeqLen;
        this.dim = dim;
        this.optimizer = optimizer;
        this.posEmbedding = new double[maxSeqLen][dim];
        this.gradPosEmbedding = new double[maxSeqLen][dim];

        for(int i = 0; i < maxSeqLen; i++) {
            initializer.initializeBiases(posEmbedding[i]);
        }
    }

    @Override
    public double[][][] forwardBatch(double[][][] data) {
        int batchSize = data.length;
        int seqLen = data[0].length;
        double[][][] output = new double[batchSize][seqLen][dim];

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int d = 0; d < dim; d++) {
                    output[b][t][d] = data[b][t][d] + posEmbedding[t][d];
                }
            }
        }
        return output;
    }

    @Override
    public double[][][] backwardBatch(double[][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        // 累积梯度到 posEmbedding
        for (int b = 0; b < batchSize; b++) {
            int seqLen = gradOutputBatch[b].length;
            for (int t = 0; t < seqLen && t < maxSeqLen; t++) {
                for (int d = 0; d < dim; d++) {
                    gradPosEmbedding[t][d] += gradOutputBatch[b][t][d];
                }
            }
        }
        return gradOutputBatch;
    }

    @Override
    public void updateParameters() {
        optimizer.applyGradients(posEmbedding, gradPosEmbedding);
        clearGrad();
    }

    private void clearGrad() {
        for (double[] grad : gradPosEmbedding) {
            Arrays.fill(grad, 0.0);
        }
    }
}
