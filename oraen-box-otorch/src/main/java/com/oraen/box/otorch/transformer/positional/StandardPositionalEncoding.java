// StandardPositionalEncoding.java
package com.oraen.box.otorch.transformer.positional;

import com.oraen.box.otorch.transformer.PositionalEncodingLayer;
import com.oraen.box.otorch.util.MathUtil;
import lombok.Getter;

@Getter
public class StandardPositionalEncoding implements PositionalEncodingLayer {

    private final int maxSeqLen;
    private final int dim;
    private final double[][] posEncoding; // [maxSeqLen, dim]

    public StandardPositionalEncoding(int maxSeqLen, int dim) {
        this.maxSeqLen = maxSeqLen;
        this.dim = dim;
        this.posEncoding = new double[maxSeqLen][dim];
        precompute();
    }

    private void precompute() {
        for (int pos = 0; pos < maxSeqLen; pos++) {
            // i递增2偶数时，使用 sin，奇数时使用 cos
            for (int i = 0; i < dim; i += 2) {
                double divTerm = Math.pow(10000.0, (2.0 * i) / dim);
                // posEncoding[pos][i] = Math.sin(pos / divTerm);
                posEncoding[pos][i] = Math.sin(pos / divTerm);
                if (i + 1 < dim) {
                    // posEncoding[pos][i + 1] = Math.cos(pos / divTerm);
                    posEncoding[pos][i + 1] = Math.cos(pos / divTerm);
                }
            }
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
                    output[b][t][d] = data[b][t][d] + posEncoding[t][d];
                }
            }
        }
        return output;
    }

    @Override
    public double[][][] backwardBatch(double[][][] gradOutputBatch) {
        // 无参数，梯度原样传回
        return gradOutputBatch;
    }
}
