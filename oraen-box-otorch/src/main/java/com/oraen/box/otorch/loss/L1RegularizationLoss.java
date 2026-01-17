package com.oraen.box.otorch.loss;

import com.oraen.box.otorch.WeightRegularizationLoss;

/**
 * L1 权重正则损失
 * L = lambda * sum(|W|)
 */
public class L1RegularizationLoss implements WeightRegularizationLoss {

    @Override
    public double compute(double[][] weight, double lambda) {
        double sum = 0.0;
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[i].length; j++) {
                sum += Math.abs(weight[i][j]);
            }
        }
        return lambda * sum;
    }
}
