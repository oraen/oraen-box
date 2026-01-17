package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.Layer;

public class PositionalEncodingLayer implements Layer<double[], double[]> {
    // forward: x + pos
    @Override
    public double[][] forwardBatch(double[][] data) {
        return new double[0][];
    }

    // backward: 梯度原样传回
    @Override
    public double[][] backwardBatch(double[][] gradOutputBatch) {
        return new double[0][];
    }

}