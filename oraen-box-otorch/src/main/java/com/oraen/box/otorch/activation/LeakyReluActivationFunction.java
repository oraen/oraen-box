package com.oraen.box.otorch.activation;


/**
 * Leaky ReLU activation function.
 */
public class LeakyReluActivationFunction extends ActivationFunctionIndependently {
    private final double alpha;

    public LeakyReluActivationFunction(double alpha) {
        this.alpha = alpha;
    }


    @Override
    public double activate(double input) {
        return input >= 0 ? input : alpha * input;
    }

    @Override
    public double derivative(double input) {
        return input >= 0 ? 1 : alpha;
    }
}
