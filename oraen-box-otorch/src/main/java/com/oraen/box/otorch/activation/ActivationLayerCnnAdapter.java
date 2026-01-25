package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;
import com.oraen.box.otorch.Layer;
import lombok.Getter;

@Getter
public class ActivationLayerCnnAdapter implements Layer<double[][][], double[][][]> {

    private final ActivationFunction activationFunction;

    /** cache forward input: [batch][channel][height][width] */
    private double[][][][] cachedInputBatch;

    public ActivationLayerCnnAdapter(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public double[][][][] forwardBatch(double[][][][] data) {
        int batchSize = data.length;
        int channels = data[0].length;
        int height = data[0][0].length;
        int width = data[0][0][0].length;

        double[][][][] output = new double[batchSize][channels][height][width];

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    // 对 width 这一维直接复用 ActivationFunction
                    output[b][c][h] =
                            activationFunction.activate(data[b][c][h]);
                }
            }
        }

        // cache input for backward
        this.cachedInputBatch = data;

        return output;
    }

    @Override
    public double[][][][] backwardBatch(double[][][][] gradOutputBatch) {
        if (cachedInputBatch == null) {
            throw new IllegalStateException(
                    "Must call forwardBatch before backwardBatch");
        }

        int batchSize = gradOutputBatch.length;
        int channels = gradOutputBatch[0].length;
        int height = gradOutputBatch[0][0].length;
        int width = gradOutputBatch[0][0][0].length;

        double[][][][] gradInputBatch =
                new double[batchSize][channels][height][width];

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {

                    // 激活函数导数：基于 forward 的输入
                    double[] derivative = activationFunction.derivative(cachedInputBatch[b][c][h]);

                    for (int w = 0; w < width; w++) {
                        gradInputBatch[b][c][h][w] =
                                gradOutputBatch[b][c][h][w] * derivative[w];
                    }
                }
            }
        }

        return gradInputBatch;
    }
}
