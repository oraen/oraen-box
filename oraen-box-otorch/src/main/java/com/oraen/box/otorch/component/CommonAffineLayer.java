package com.oraen.box.otorch.component;

import com.oraen.box.otorch.AffineLayer;
import com.oraen.box.otorch.GradientsMsg;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.GradOptimizer;
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


    public CommonAffineLayer(int inputDim, int outputDim, ParamInitializer paramInitializer, GradOptimizer gradOptimizer) {
        super(inputDim, outputDim);
        this.paramInitializer = paramInitializer;
        this.gradOptimizer = gradOptimizer;
        paramInitializer.initializeWeights(this.weight);
        paramInitializer.initializeBiases(this.bias);
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
        if(lastForwardInput == null || batchSize != lastForwardInput.length) {
            throw new IllegalArgumentException("gradOutput batch size does not match lastInput batch size.");
        }

        double[][] gradInput = new double[batchSize][inputDim];


        for (int i = 0; i < batchSize; i++) {
            double[] gradInputItem = backward(gradOutputBatch[i], lastForwardInput[i]);
            gradInput[i] = gradInputItem;

        }

        this.lastGradOutputBatch = gradOutputBatch;

        return gradInput;
    }



    @Override
    public void updateParameters() {
        GradientsMsg gradientsMsg = getGradientsMsg(lastGradOutputBatch);
        gradOptimizer.applyGradients(this.weight, this.bias, gradientsMsg);

    }

    private GradientsMsg getGradientsMsg(double[][] outputGradients) {
        int batchSIze = outputGradients.length;
        GradientsMsg re = new GradientsMsg(new double[outputDim][inputDim], new double[outputDim]);

        for(int i = 0; i < batchSIze; i ++) {
            GradientsMsg layerGradients = getGradientsMsg(outputGradients[i], lastForwardInput[i]);
            // 累加参数梯度
            re.plus(layerGradients);
        }
        return re;
    }


    private GradientsMsg getGradientsMsg(double[] gradOutput, double[] input) {
        double[][] gradWeight = new double[outputDim][inputDim];
        double[] gradBias = new double[outputDim];

        // gradWeight = gradOutput ⊗ lastInput
        for (int j = 0; j < outputDim; j++) {
            for (int k = 0; k < inputDim; k++) {
                gradWeight[j][k] = gradOutput[j] * input[k];
            }
            gradBias[j] = gradOutput[j];
        }

        return new GradientsMsg(gradWeight, gradBias);
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

    private double[] backward(double[] gradOutput, double[] input) {
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
