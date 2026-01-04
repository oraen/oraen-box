package com.oraen.box.otorch;

public interface Affine extends ComputationalNode {

    AffineGradientsMsg getGradientsMsg(OTorchContext oTorchContext, double[][] outputGradients);

    void applyGradients(OTorchContext oTorchContext, AffineGradientsMsg gradients);

}
