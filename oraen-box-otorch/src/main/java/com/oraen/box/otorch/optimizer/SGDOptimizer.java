package com.oraen.box.otorch.optimizer;

import com.oraen.box.otorch.GradOptimizer;
import lombok.Data;

@Data
public class SGDOptimizer implements GradOptimizer {

    private double learningRate;

    public SGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public SGDOptimizer() {
        this(0.01);
    }

    @Override
    public void applyGradients(double[][] weight, double[][] gradWeight) {
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[i].length; j++) {
                weight[i][j] -= learningRate * gradWeight[i][j];
            }
        }
    }

    @Override
    public void applyGradients(double[] bias, double[] gradBias) {
        for (int i = 0; i < bias.length; i++) {
            bias[i] -= learningRate * gradBias[i];
        }
    }
}
