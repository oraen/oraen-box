package com.oraen.box.otorch;


public interface ComputationalNode {

    double[][] forwardBatch(OTorchContext oTorchContext, double[][] data);

    double[][] backwardBatch(OTorchContext oTorchContext, double[][] gradOutputBatch);
}
