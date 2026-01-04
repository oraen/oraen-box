package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.LossFunction;

/**
 * 平滑 L1 损失
 */
public class SmoothL1LossFunction implements LossFunction {
    private final double beta;

    public SmoothL1LossFunction(double beta) {
        this.beta = beta;
    }

    @Override
    public double computeLoss(double[] predicted, double[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual arrays must have the same length.");
        }

        double totalLoss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = Math.abs(predicted[i] - actual[i]);
            if (diff < beta) {
                totalLoss += 0.5 * diff * diff / beta;
            } else {
                totalLoss += diff - 0.5 * beta;
            }
        }

        return totalLoss / predicted.length;
    }
}
