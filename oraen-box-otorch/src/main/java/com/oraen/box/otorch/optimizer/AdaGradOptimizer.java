package com.oraen.box.otorch.optimizer;

import com.oraen.box.otorch.GradOptimizer;
import lombok.Getter;
import lombok.Setter;

/**
 * AdaGrad 自适应梯度优化器
 */
@Getter
public class AdaGradOptimizer implements GradOptimizer {

    /**
     * //学习率 learningRate 学习率
     * //控制参数更新步长
     */
    @Setter
    private double learningRate;
    private final int outputDim;
    private final int inputDim;

    private final double eps = 1e-8;

    /**
     * //权重梯度平方累计量 hW
     * //Accumulator 累计器
     * //用于记录“每个权重位置上”历史梯度强度
     * //最终会压小梯度大的维度，放大梯度小的维度
     */
    private final double[][] hW;

    /**
     * //偏置梯度平方累计量 hB
     * //用于缩放 bias 的学习速度
     */
    private final double[] hB;

    /**
     * //构造函数
     * //根据网络层维度直接分配状态
     */
    public AdaGradOptimizer(int outputDim, int inputDim) {
        this(outputDim, inputDim, 0.01);
    }

    /**
     * //完整构造
     * //lr: 学习率
     */
    public AdaGradOptimizer(int outputDim,
                            int inputDim,
                            double learningRate) {

        this.learningRate = learningRate;
        this.outputDim = outputDim;
        this.inputDim = inputDim;

        // 分配累计器
        this.hW = new double[outputDim][inputDim];
        this.hB = new double[outputDim];
    }



    @Override
    public void applyGradients(double[][] weight, double[][] gradWeight) {
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                // 梯度平方累计（与方向无关）
                hW[i][j] += gradWeight[i][j] * gradWeight[i][j];
                // 按历史强度缩放
                weight[i][j] -= learningRate * gradWeight[i][j] / Math.sqrt(hW[i][j] + eps);
            }
        }
    }

    @Override
    public void applyGradients(double[] bias, double[] gradBias) {
        for (int i = 0; i < outputDim; i++) {
            // bias 的梯度平方累计
            hB[i] += gradBias[i] * gradBias[i];
            bias[i] -= learningRate * gradBias[i] / Math.sqrt(hB[i] + eps);
        }
    }
}
