package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;
import com.oraen.box.otorch.Layer;
import lombok.Getter;

@Getter
public class ActivationLayerAdapter implements Layer<double[], double[]> {

    private final ActivationFunctionIndependently activationFunction;

    /** cache forward input: shape = [batch][dim] */
    private double[][] cachedInputBatch;

    public ActivationLayerAdapter(ActivationFunctionIndependently activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public double[][] forwardBatch(double[][] data) {
        int batchSize = data.length;
        double[][] output = new double[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            output[i] = activationFunction.activate(data[i]);
        }

        // cache input for backward
        this.cachedInputBatch = data;

        return output;
    }

    @Override
    public double[][] backwardBatch(double[][] gradOutputBatch) {


        int batchSize = gradOutputBatch.length;
        int inputDim = cachedInputBatch[0].length;
        double[][] gradInputBatch = new double[batchSize][inputDim];

        for (int i = 0; i < batchSize; i++) {
            double[] derivative = activationFunction.derivative(cachedInputBatch[i]);
            for(int j = 0; j < inputDim; j++) {
                gradInputBatch[i][j] = gradOutputBatch[i][j] * derivative[j];
            }
        }

        return gradInputBatch;

    }
}