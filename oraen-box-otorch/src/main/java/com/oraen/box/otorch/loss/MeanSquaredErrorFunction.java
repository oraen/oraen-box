package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.LossFunction;

/**
 * 均方误差
 */
public class MeanSquaredErrorFunction implements LossFunction {


    @Override
    public double computeLoss(double[] predicted, double[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual arrays must have the same length.");
        }
        double sumSquaredError = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double error = predicted[i] - actual[i];
            sumSquaredError += error * error;
        }
        return sumSquaredError / predicted.length;
    }
}
