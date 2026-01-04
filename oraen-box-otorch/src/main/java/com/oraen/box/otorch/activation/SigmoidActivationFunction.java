package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;

/**
 * Sigmoid activation function.
 */
public class SigmoidActivationFunction extends ActivationFunctionIndependently  {


    @Override
    public double activate(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
        // 方法1：直接使用sigmoid函数值计算（推荐，数值稳定）
        double sigmoid = activate(input);
        return sigmoid * (1 - sigmoid);

        // 方法2：使用数学公式 exp(-x) / (1 + exp(-x))^2
//        double expNegX = Math.exp(-input);
//        double denominator = 1 + expNegX;
//        return expNegX / (denominator * denominator);
    }
}
