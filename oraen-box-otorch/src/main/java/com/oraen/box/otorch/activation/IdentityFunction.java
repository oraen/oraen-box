package com.oraen.box.otorch.activation;

public class IdentityFunction extends ActivationFunctionIndependently  {

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double derivative(double input) {
        return 1;
    }
}
