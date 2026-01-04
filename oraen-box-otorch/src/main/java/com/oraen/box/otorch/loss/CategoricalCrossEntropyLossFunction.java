package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.LossFunction;

/**
 * 分类交叉熵损失函数
 */
public class CategoricalCrossEntropyLossFunction implements LossFunction {

    private static final double EPS = 1e-12;
    @Override
    public double computeLoss(double[] predicted, double[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual arrays must have the same length.");
        }

        double totalLoss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            // Only consider the true class
            if (actual[i] == 1.0) {
                // Adding a small constant to avoid log(0)
                totalLoss -= Math.log(predicted[i] + EPS);
            }
        }

        return totalLoss;
    }
}
