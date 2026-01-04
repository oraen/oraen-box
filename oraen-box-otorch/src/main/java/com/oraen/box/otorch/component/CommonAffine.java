package com.oraen.box.otorch.component;

import com.oraen.box.otorch.Affine;
import com.oraen.box.otorch.AffineGradientsMsg;
import com.oraen.box.otorch.OTorchContext;
import com.oraen.box.otorch.ParamInitializer;
import lombok.Getter;


@Getter
public class CommonAffine implements Affine {
    private final int inputDim;
    private final int outputDim;
    private final double[][] weight;
    private final double[] bias;
    // forward 时缓存
    private double[][] lastInput;


    public CommonAffine(int inputDim, int outputDim, ParamInitializer paramInitializer) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.weight = new double[outputDim][inputDim];
        this.bias = new double[outputDim];
        paramInitializer.initializeWeights(this.weight);
        paramInitializer.initializeBiases(this.bias);
    }



    /**
     * 前向传播
     * @param oTorchContext
     * @param data
     * @return
     */
    public double[][] forwardBatch(OTorchContext oTorchContext, double[][] data) {
        int dataSize = data.length;

        double[][] out = new double[dataSize][];

        for(int i = 0; i < dataSize; i++) {
            out[i] = forward(oTorchContext, data[i]);
        }

        this.lastInput = data;

        return out;
    }



    public double[][] backwardBatch(OTorchContext oTorchContext, double[][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        if(lastInput == null || batchSize != lastInput.length) {
            throw new IllegalArgumentException("gradOutput batch size does not match lastInput batch size.");
        }

        double[][] gradInput = new double[batchSize][inputDim];


        for (int i = 0; i < batchSize; i++) {
            double[] gradInputItem = backward(oTorchContext, gradOutputBatch[i], lastInput[i]);
            gradInput[i] = gradInputItem;

        }

        return gradInput;
    }


    @Override
    public AffineGradientsMsg getGradientsMsg(OTorchContext oTorchContext, double[][] outputGradients) {
        int batchSIze = outputGradients.length;
        double[][] gradWeight = new double[outputDim][inputDim];
        double[] gradBias = new double[outputDim];

        for(int i = 0; i < batchSIze; i ++) {
            AffineGradientsMsg layerGradients = getGradientsMsg(oTorchContext, outputGradients[i], lastInput[i]);
            // 累加参数梯度
            for (int j = 0; j < outputDim; j++) {
                gradBias[j] += layerGradients.getGradBiases()[j];
                for (int k = 0; k < inputDim; k++) {
                    gradWeight[j][k] += layerGradients.getGradWeights()[j][k];
                }
            }
        }

        return new AffineGradientsMsg(gradWeight, gradBias);
    }

    @Override
    public void applyGradients(OTorchContext oTorchContext, AffineGradientsMsg gradients) {

    }


    private AffineGradientsMsg getGradientsMsg(OTorchContext oTorchContext, double[] gradOutput, double[] input) {
        double[][] gradWeight = new double[outputDim][inputDim];
        double[] gradBias = new double[outputDim];

        // gradWeight = gradOutput ⊗ lastInput
        for (int j = 0; j < outputDim; j++) {
            for (int k = 0; k < inputDim; k++) {
                gradWeight[j][k] = gradOutput[j] * input[k];
            }
            gradBias[j] = gradOutput[j];
        }

        return new AffineGradientsMsg(gradWeight, gradBias);
    }


    /**
     * 前向传播
     * @param oTorchContext
     * @param data
     * @return
     */
    private double[] forward(OTorchContext oTorchContext, double[] data) {
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

    private double[] backward(OTorchContext oTorchContext, double[] gradOutput, double[] input) {
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
