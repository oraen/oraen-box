package com.oraen.box.otorch;

/**
 * 权重正则项损失（仅用于数值计算）
 */
public interface WeightRegularizationLoss {

    /**
     * @param weight 权重矩阵
     * @param lambda 正则强度
     * @return 正则损失值
     */
    double compute(double[][] weight, double lambda);
}
