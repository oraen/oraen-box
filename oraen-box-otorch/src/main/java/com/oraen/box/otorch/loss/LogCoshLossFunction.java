package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.LossFunction;

/**
 * 对数双曲余弦损失函数
 */
public class LogCoshLossFunction implements LossFunction {
    @Override
    public double computeLoss(double[] predicted, double[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual arrays must have the same length.");
        }

        double totalLoss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double error = predicted[i] - actual[i];
            totalLoss += Math.log(Math.cosh(error));
        }

        return totalLoss / predicted.length;
    }
}
