package com.oraen.box.otorch;

public interface LossFunction {
    double computeLoss(double[] predicted, double[] actual);
}
