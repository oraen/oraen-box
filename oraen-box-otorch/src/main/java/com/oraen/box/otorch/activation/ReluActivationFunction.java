package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;

public class ReluActivationFunction extends ActivationFunctionIndependently  {


    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : 0;
    }
}
