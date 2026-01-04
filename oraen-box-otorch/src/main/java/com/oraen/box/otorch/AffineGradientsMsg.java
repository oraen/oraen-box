package com.oraen.box.otorch;

import lombok.Data;

@Data
public class AffineGradientsMsg {
    private final double[][] gradWeights;  // 权重梯度
    private final double[] gradBiases;     // 偏置梯度

    public AffineGradientsMsg( double[][] gradWeights, double[] gradBiases) {
        this.gradWeights = gradWeights;
        this.gradBiases = gradBiases;
    }
}