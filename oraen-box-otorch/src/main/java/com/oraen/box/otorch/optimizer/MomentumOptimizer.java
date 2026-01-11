package com.oraen.box.otorch.optimizer;

import com.oraen.box.otorch.GradOptimizer;
import lombok.Getter;
import lombok.Setter;

@Getter
public class MomentumOptimizer implements GradOptimizer {

    private final int outputDim;
    private final int inputDim;

    @Setter
    private double learningRate;

    //0.0 时候就退化成SGD
    @Setter
    private double momentum;
    private final double[][] vW;
    private final double[] vB;

    //因为需要记录动量，所以需要在构造函数中初始化动量变量的维度，并且初始化vw，vb，不可被混用
    public MomentumOptimizer(int outputDim, int inputDim, double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.outputDim = outputDim;
        this.inputDim = inputDim;
        this.vW = new double[outputDim][inputDim];
        this.vB = new double[outputDim];
    }

    public MomentumOptimizer(int outputDim, int inputDim) {
        this(outputDim, inputDim, 0.01, 0.9);
    }

    @Override
    public void applyGradients(double[][] weight, double[][] gradWeight) {
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                vW[i][j] = momentum * vW[i][j] + gradWeight[i][j];
                weight[i][j] -= learningRate * vW[i][j];
            }
        }
    }

    @Override
    public void applyGradients(double[] bias, double[] gradBias) {
        for (int i = 0; i < outputDim; i++) {
            vB[i] = momentum * vB[i] + gradBias[i];
            bias[i] -= learningRate * vB[i];
        }
    }
}