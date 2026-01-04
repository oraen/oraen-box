package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;

public class TanhActivationFunction extends ActivationFunctionIndependently  {

    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derivative(double input) {
        // tanh'(x) = 1 - tanh(x)^2
        double t = Math.tanh(input);
        return 1.0 - t * t;
    }
}
