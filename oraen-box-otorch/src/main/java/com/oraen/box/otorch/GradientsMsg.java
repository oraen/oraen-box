package com.oraen.box.otorch;

import lombok.Data;

@Data
public class GradientsMsg {
    private final double[][] gradWeights;  // 权重梯度
    private final double[] gradBiases;     // 偏置梯度

    public GradientsMsg(double[][] gradWeights, double[] gradBiases) {
        this.gradWeights = gradWeights;
        this.gradBiases = gradBiases;
    }

    public void plus(GradientsMsg other) {
        for (int i = 0; i < gradWeights.length; i++) {
            for (int j = 0; j < gradWeights[i].length; j++) {
                this.gradWeights[i][j] += other.gradWeights[i][j];
            }

            this.gradBiases[i] += other.gradBiases[i];
        }
    }
}