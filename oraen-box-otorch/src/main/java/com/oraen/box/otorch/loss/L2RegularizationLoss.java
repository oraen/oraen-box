package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.WeightRegularizationLoss;

/**
 * L2 权重正则损失
 * L = 0.5 * lambda * sum(W^2)
 */
public class L2RegularizationLoss implements WeightRegularizationLoss {

    @Override
    public double compute(double[][] weight, double lambda) {
        double sum = 0.0;
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[i].length; j++) {
                double w = weight[i][j];
                sum += w * w;
            }
        }
        return 0.5 * lambda * sum;
    }
}
