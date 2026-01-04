package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;

public abstract class ActivationFunctionIndependently implements ActivationFunction {

    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = activate(input[i]);
        }

        return output;
    }

    @Override
    public double[] derivative(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = derivative(input[i]);
        }

        return output;
    }

    public abstract double activate(double input);

    public abstract double derivative(double input);

}
