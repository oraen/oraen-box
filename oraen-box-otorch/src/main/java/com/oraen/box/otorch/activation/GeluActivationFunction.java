package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;

/**
 * GELU
 */
public class GeluActivationFunction extends ActivationFunctionIndependently {

    @Override
    public double activate(double input) {
        // GELU activation function approximation
        return 0.5 * input * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715 * Math.pow(input, 3))));
    }

    @Override
    public double derivative(double input) {
        double x3 = input * input * input;
        double inner = 0.0356774 * x3 + 0.797885 * input; // 优化后的常数

        double tanhVal = Math.tanh(inner);
        double sech2 = 1 - tanhVal * tanhVal; // sech²(x) = 1 - tanh²(x)

        return 0.5 * tanhVal +
                (0.0535161 * x3 + 0.398942 * input) * sech2 + 0.5;
    }


}
