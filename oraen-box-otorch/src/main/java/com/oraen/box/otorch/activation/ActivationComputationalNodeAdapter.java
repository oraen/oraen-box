package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;
import com.oraen.box.otorch.ComputationalNode;
import com.oraen.box.otorch.OTorchContext;

public class ActivationComputationalNodeAdapter implements ComputationalNode {

    private final ActivationFunction activationFunction;

    /** cache forward input: shape = [batch][dim] */
    private double[][] cachedInputBatch;

    public ActivationComputationalNodeAdapter(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public double[][] forwardBatch(OTorchContext oTorchContext, double[][] data) {
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
    public double[][] backwardBatch(OTorchContext oTorchContext,  double[][] gradOutputBatch) {

        if (cachedInputBatch == null) {
            throw new IllegalStateException(
                    "forwardBatch must be called before backwardBatch"
            );
        }

        int batchSize = gradOutputBatch.length;
        double[][] gradInputBatch = new double[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            gradInputBatch[i] = activationFunction.derivative(cachedInputBatch[i]);
        }

        return gradInputBatch;
    }
}