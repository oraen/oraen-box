package com.oraen.box.otorch;

public interface GradOptimizer {

    void applyGradients(double[][] weight, double[] bias, GradientsMsg gradientsMsg);

}
