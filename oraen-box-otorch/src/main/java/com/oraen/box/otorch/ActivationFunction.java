package com.oraen.box.otorch;

public interface ActivationFunction {

    double[] activate(double[] input);

    double[] derivative(double[] inputs);
}
