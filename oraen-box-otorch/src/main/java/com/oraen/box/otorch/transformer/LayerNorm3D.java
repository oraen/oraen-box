package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.convert.AdaptiveMergeFirstDimsConvert;
import lombok.Getter;

public class LayerNorm3D implements Layer<double[][], double[][]>, Learnable {

    @Getter
    private final LayerNorm layerNorm;

    private final AdaptiveMergeFirstDimsConvert adaptiveMergeFirstDimsConvert = new AdaptiveMergeFirstDimsConvert();

    public LayerNorm3D(int dim, GradOptimizer gammaOptimizer, GradOptimizer betaOptimizer, ParamInitializer gammaParamInitializer, ParamInitializer betaParamInitializer) {
        this.layerNorm = new LayerNorm(dim, gammaOptimizer, betaOptimizer, gammaParamInitializer, betaParamInitializer);
    }

    public LayerNorm3D(LayerNorm layerNorm) {
        this.layerNorm = layerNorm;
    }


    @Override
    public void updateParameters() {
        layerNorm.updateParameters();
    }

    @Override
    public double[][][] forwardBatch(double[][][] data) {
        double[][] dataFlatten = adaptiveMergeFirstDimsConvert.forwardBatch(data);
        double[][] normOut = layerNorm.forwardBatch(dataFlatten);
        return adaptiveMergeFirstDimsConvert.backwardBatch(normOut);
    }

    @Override
    public double[][][] backwardBatch(double[][][] gradOutputBatch) {
        double[][] dataFlatten = adaptiveMergeFirstDimsConvert.forwardBatch(gradOutputBatch);
        double[][] normOut = layerNorm.backwardBatch(dataFlatten);
        return adaptiveMergeFirstDimsConvert.backwardBatch(normOut);
    }
}
