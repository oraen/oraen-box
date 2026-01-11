package com.oraen.box.otorch.component;

import com.oraen.box.otorch.AffineLayer;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.GradOptimizer;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;


@Getter
public class CommonAffineLayer extends AffineLayer {
    // forward 时缓存
    private double[][] lastForwardInput;

    // backwardBatch 时缓存
    private double[][] lastGradOutputBatch;

    private final ParamInitializer paramInitializer;

    @Setter
    private GradOptimizer gradOptimizer;

    private double[][] gradWeight;

    private double[] gradBias;

    public CommonAffineLayer(int inputDim, int outputDim, ParamInitializer paramInitializer, GradOptimizer gradOptimizer) {
        super(inputDim, outputDim);
        this.paramInitializer = paramInitializer;
        this.gradOptimizer = gradOptimizer;
        paramInitializer.initializeWeights(this.weight);
        paramInitializer.initializeBiases(this.bias);
        this.gradWeight = new double[outputDim][inputDim];
        this.gradBias = new double[outputDim];
    }

    /**
     * 前向传播
     * @param data
     * @return
     */
    public double[][] forwardBatch(double[][] data) {
        int dataSize = data.length;

        double[][] out = new double[dataSize][];

        for(int i = 0; i < dataSize; i++) {
            out[i] = forward(data[i]);
        }

        this.lastForwardInput = data;

        return out;
    }



    public double[][] backwardBatch( double[][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;

        double[][] gradInput = new double[batchSize][inputDim];


        for (int i = 0; i < batchSize; i++) {
            double[] gradInputItem = backward(gradOutputBatch[i]);
            gradInput[i] = gradInputItem;
        }

        this.lastGradOutputBatch = gradOutputBatch;
        updateGradientsMsg(gradOutputBatch);
        return gradInput;
    }



    @Override
    public void updateParameters() {
        gradOptimizer.applyGradients(weight, bias, gradWeight, gradBias);
        resetGradients();
    }

    private void resetGradients() {
        this.gradWeight = new double[outputDim][inputDim];
        this.gradBias = new double[outputDim];
    }

    private void updateGradientsMsg(double[][] outputGradients) {
        int batchSIze = outputGradients.length;
        for(int i = 0; i < batchSIze; i ++) {
            updateGradientsMsg(outputGradients[i], lastForwardInput[i]);
        }
    }

    private void updateGradientsMsg(double[] gradOutput, double[] input) {
        // gradWeight = gradOutput ⊗ lastInput
        for (int j = 0; j < outputDim; j++) {
            for (int k = 0; k < inputDim; k++) {
                gradWeight[j][k] += gradOutput[j] * input[k];
            }
            gradBias[j] += gradOutput[j];
        }
    }


    /**
     * 前向传播
     * @param data
     * @return
     */
    private double[] forward(double[] data) {
        int dataDim = data.length;
        if(dataDim != inputDim) {
            throw new IllegalArgumentException("Input data dimension does not match affine layer input dimension.");
        }

        double[] out = new double[this.outputDim];

        for(int i = 0; i < outputDim; i ++){
            double sum = 0.0;
            for(int j = 0; j < inputDim; j ++){
                sum += data[j] * this.weight[i][j];
            }
            sum += this.bias[i];
            out[i] = sum;
        }

        return out;
    }

    private double[] backward(double[] gradOutput) {
        if (gradOutput.length != outputDim) {
            throw new IllegalArgumentException("gradOutput dimension mismatch");
        }

        double[] gradInput = new double[inputDim];

        // gradInput = Wᵀ · gradOutput
        for (int k = 0; k < inputDim; k++) {
            double sum = 0.0;
            for (int j = 0; j < outputDim; j++) {
                sum += weight[j][k] * gradOutput[j];
            }
            gradInput[k] = sum;
        }

        return gradInput;
    }
}
