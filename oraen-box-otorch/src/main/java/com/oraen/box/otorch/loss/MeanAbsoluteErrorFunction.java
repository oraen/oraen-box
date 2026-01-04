package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.LossFunction;

/**
 * 平均绝对误差
 */
public class MeanAbsoluteErrorFunction implements LossFunction {
    @Override
    public double computeLoss(double[] predicted, double[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual arrays must have the same length.");
        }

        double totalError = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            totalError += Math.abs(predicted[i] - actual[i]);
        }

        return totalError / predicted.length;
    }
}
