package com.oraen.box.otorch;

public abstract class AffineLayer implements Layer<double[], double[]>,Learnable {

    protected final int inputDim;
    protected final int outputDim;
    protected final double[][] weight;
    protected final double[] bias;

    public AffineLayer(int inputDim, int outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.weight = new double[outputDim][inputDim];
        this.bias = new double[outputDim];
    }

}
