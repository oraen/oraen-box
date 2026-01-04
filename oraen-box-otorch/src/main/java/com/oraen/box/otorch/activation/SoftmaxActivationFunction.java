package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;
import lombok.Getter;
import lombok.Setter;

/**
 * Softmax activation function implementation.
 * Transforms the input array into a probability distribution.
 */
public class SoftmaxActivationFunction implements ActivationFunction {


    /** Upstream gradient: dL / dY
     * -- SETTER --
     * Inject upstream gradient manually
     */
    @Setter
    @Getter
    private double[] upstreamGradient;

    /** Cached output of softmax (y) */
    private double[] cachedOutput;

    /** Empty constructor (derivative() will fail if gradient not set) */
    public SoftmaxActivationFunction() {
    }

    /** Constructor with upstream gradient */
    public SoftmaxActivationFunction(double[] upstreamGradient) {
        this.upstreamGradient = upstreamGradient;
    }


    @Override
    public double[] activate(double[] input) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : input) {
            if (v > max) {
                max = v;
            }
        }

        double sum = 0.0;
        double[] expValues = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            expValues[i] = Math.exp(input[i] - max);
            sum += expValues[i];
        }

        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = expValues[i] / sum;
        }

        this.cachedOutput = output;
        return output;
    }

    @Override
    public double[] derivative(double[] inputs) {
        if (upstreamGradient == null) {
            throw new IllegalStateException(
                    "Upstream gradient (dL/dY) is not set for SoftmaxActivationFunction"
            );
        }

        if (cachedOutput == null) {
            throw new IllegalStateException(
                    "activate() must be called before derivative()"
            );
        }

        if (upstreamGradient.length != cachedOutput.length) {
            throw new IllegalArgumentException(
                    "Gradient size does not match softmax output size"
            );
        }

        int n = cachedOutput.length;
        double[] dLdX = new double[n];

        // compute sum_j (dL/dY_j * Y_j)
        double dot = 0.0;
        for (int j = 0; j < n; j++) {
            dot += upstreamGradient[j] * cachedOutput[j];
        }

        // dL/dX_i = Y_i * (dL/dY_i - dot)
        for (int i = 0; i < n; i++) {
            dLdX[i] = cachedOutput[i] * (upstreamGradient[i] - dot);
        }

        return dLdX;
    }
}
