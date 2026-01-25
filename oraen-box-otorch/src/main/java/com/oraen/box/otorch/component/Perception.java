package com.oraen.box.otorch.component;

import com.oraen.box.otorch.ActivationFunction;
import com.oraen.box.otorch.AffineLayer;
import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.activation.ActivationFunctionIndependently;
import com.oraen.box.otorch.activation.ActivationLayerAdapter;

/**
 * 感知器
 */
public class Perception implements Layer<double[], double[]>, Learnable {

    AffineLayer affineLayer;

    ActivationLayerAdapter activationLayerAdapter;

    public Perception(AffineLayer affineLayer, ActivationFunctionIndependently activationFunction) {
        this.affineLayer = affineLayer;
        this.activationLayerAdapter = new ActivationLayerAdapter(activationFunction);
    }

    @Override
    public double[][] forwardBatch(double[][] data) {

        double[][] data1 = affineLayer.forwardBatch(data);
        return activationLayerAdapter.forwardBatch(data1);
    }

    @Override
    public double[][] backwardBatch(double[][] gradOutputBatch) {
        double[][] data1 = activationLayerAdapter.backwardBatch(gradOutputBatch);
        return affineLayer.backwardBatch(data1);

    }

    @Override
    public void updateParameters() {
        affineLayer.updateParameters();
    }
}
