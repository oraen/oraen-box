package com.oraen.box.otorch.optimizer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.GradientsMsg;
import lombok.Getter;
import lombok.Setter;

@Getter
public class RMSPropOptimizer implements GradOptimizer {

    private final int outputDim;
    private final int inputDim;

    @Setter
    private double learningRate;
    @Setter
    private double beta;

    private final double eps = 1e-8;

    //用于估计这个权重维度上梯度的历史的大小/强度,对于同一个位置上的权重历史上的梯度，最终梯度大的维度会被压小，梯度小的维度会被相对放大
    private final double[][] sW;
    private final double[]   sB;

    public RMSPropOptimizer(int outputDim, int inputDim) {
        this(outputDim, inputDim, 0.001, 0.9);
    }

    public RMSPropOptimizer(int outputDim, int inputDim, double learningRate, double beta) {
        this.learningRate = learningRate;
        this.outputDim = outputDim;
        this.inputDim = inputDim;
        this.sW = new double[outputDim][inputDim];
        this.sB = new double[outputDim];
        this.beta = beta;
    }


    @Override
    public void applyGradients(double[][] weight, double[] bias, GradientsMsg gradientsMsg) {
        double[][] gradW = gradientsMsg.getGradWeights();
        double[] gradB = gradientsMsg.getGradBiases();
        for (int i = 0; i < outputDim; i++) {
            // 权重
            for (int j = 0; j < inputDim; j++) {
                //先块后慢，如果beta是0.9. 第一次训练就是1/SQRT 0.1 倍速，大约3.16倍速，后面会越来越慢
                sW[i][j] = beta * sW[i][j] + (1 - beta) * gradW[i][j] * gradW[i][j];
                weight[i][j] -= learningRate * gradW[i][j] / Math.sqrt(sW[i][j] + eps);
            }

            // 偏置
            sB[i] = beta * sB[i] + (1 - beta) * gradB[i] * gradB[i];
            bias[i] -= learningRate * gradB[i] / Math.sqrt(sB[i] + eps);
        }
    }
}