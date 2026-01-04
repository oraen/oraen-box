package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.LossFunction;

/**
 * 负对数似然损失函数
 */
public class NegativeLogLikelihoodFunction implements LossFunction {

    private static final double EPS = 1e-12;

    @Override
    public double computeLoss(double[] predicted, double[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual arrays must have the same length.");
        }

        double loss = 0.0;

        for (int i = 0; i < predicted.length; i++) {
            double y = actual[i];
            double p = predicted[i];

            if (y < 0.0 || y > 1.0) {
                throw new IllegalArgumentException("Actual values must be in [0, 1].");
            }

            //Adding a small constant to avoid log(0)
            loss -= y * Math.log(p + EPS);
        }


        return loss;
    }
}
