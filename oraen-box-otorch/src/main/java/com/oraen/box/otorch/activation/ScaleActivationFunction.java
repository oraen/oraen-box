package com.oraen.box.otorch.activation;

import com.oraen.box.otorch.ActivationFunction;
import lombok.Getter;
import lombok.Setter;

public class ScaleActivationFunction extends ActivationFunctionIndependently {

    private final double scale;

    public ScaleActivationFunction (double scale) {
        this.scale = scale;
    }

    @Override
    public double activate(double input) {
        return input * scale;
    }

    @Override
    public double derivative(double input) {
        return scale;
    }
}
