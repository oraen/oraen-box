package com.oraen.box.otorch;

public interface GradOptimizer {

    void applyGradients(double[][] weight, double[][] gradWeight);

    void applyGradients(double[] bias, double[] gradBias);

    default void applyGradients(double[][] weight, double[] bias, double[][] gradWeight, double[] gradBias){
        applyGradients(weight, gradWeight);
        applyGradients(bias, gradBias);
    }

}
