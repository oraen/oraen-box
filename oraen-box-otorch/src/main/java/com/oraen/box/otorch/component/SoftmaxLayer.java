package com.oraen.box.otorch.component;

import com.oraen.box.otorch.Layer;

/**
 * A standalone Softmax layer that operates on a batch of vectors.
 * Each row in the input batch is treated as an independent sample,
 * and softmax is applied along the last dimension (i.e., per-sample).
 */
public class SoftmaxLayer implements Layer<double[], double[]> {

    // Cache the softmax output (y) for backward pass
    private double[][] cachedOutput;

    @Override
    public double[][] forwardBatch(double[][] data) {
        int batchSize = data.length;
        double[][] output = new double[batchSize][];

        // Apply softmax to each sample independently
        for (int i = 0; i < batchSize; i++) {
            output[i] = forward0(data[i]);
        }

        this.cachedOutput = output; // cache for backward
        return output;
    }

    @Override
    public double[][] backwardBatch(double[][] gradOutputBatch) {
        int batchSize = cachedOutput.length;
        int dim = cachedOutput[0].length;
        double[][] gradInput = new double[batchSize][dim];

        for (int i = 0; i < batchSize; i++) {

            // Compute dot = sum_j (gradOutput_j * y_j)
            double dot = 0.0;
            for (int j = 0; j < dim; j++) {
                dot += gradOutputBatch[i][j] * cachedOutput[i][j];
            }

            // gradInput_i = y_i * (gradOutput_i - dot)
            for (int j = 0; j < dim; j++) {
                gradInput[i][j] = cachedOutput[i][j] * (gradOutputBatch[i][j] - dot);
            }
        }

        return gradInput;
    }


    private double[] forward0(double[] data) {
        int len = data.length;
        double[] output = new double[len];

        // Numerical stability: subtract max
        double max = data[0];
        for (int i = 1; i < len; i++) {
            if (data[i] > max) max = data[i];
        }

        double sum = 0.0;
        double[] exps = new double[len];
        for (int i = 0; i < len; i++) {
            exps[i] = Math.exp(data[i] - max);
            sum += exps[i];
        }

        for (int i = 0; i < len; i++) {
            output[i] = exps[i] / sum;
        }

        return output;


    }
}
