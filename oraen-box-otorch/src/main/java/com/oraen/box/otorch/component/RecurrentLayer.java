package com.oraen.box.otorch.component;

import com.oraen.box.otorch.*;
import lombok.Data;


/**
 * RNN层（填充方案）
 * 输入形状：[batch_size, seq_len, input_dim]
 * 输出形状：[batch_size, seq_len, hidden_dim]
 * 支持可变长度（通过seqLen参数）
 */
@Data
public class RecurrentLayer implements Layer<double[][], double[][]>, Learnable {

    // 输入到隐藏层权重，将输入转换到隐藏空间
    private final double[][] wxh;
    // 隐藏层递归权重，记忆跨时间步信息
    private final double[][] whh;
    // 隐藏层偏置
    private final double[] bh;

    // 输入特征维度
    private final int inputDim;
    // 隐藏状态维度，决定记忆容量
    private final int outputDim;

    // 梯度优化器
    // 权重更新策略
    private GradOptimizer gradOptimizer;

    private ActivationFunction activationFunction;

    // 前向传播缓存（用于反向传播）
    // 各时间步输入，计算Wxh梯度
    private double[][][] cachedInputs;
    // 各时间步隐藏状态，计算Whh梯度
    private double[][][] cachedOutputs;

    // 实际序列长度，处理可变长度序列
    private int[] cachedSeqLens;

    // 梯度累加
    // Wxh梯度累加和
    private double[][] gradWxh;
    // Whh梯度累加和
    private double[][] gradWhh;
    // bh梯度累加和
    private double[] gradBh;



    public RecurrentLayer(int inputDim, int outputDim, ParamInitializer paramInitializer, GradOptimizer gradOptimizer, ActivationFunction activationFunction) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.gradOptimizer = gradOptimizer;
        this.activationFunction = activationFunction;

        // 初始化权重
        this.wxh = new double[outputDim][inputDim];
        this.whh = new double[outputDim][outputDim];
        this.bh = new double[outputDim];

        paramInitializer.initializeWeights(wxh);
        paramInitializer.initializeWeights(whh);
        paramInitializer.initializeBiases(bh);

        resetGradients();
    }

    @Override
    public double[][][] forwardBatch(double[][][] batchData) {
        this.cachedInputs = batchData;
        int batchSize = batchData.length;
        double[][][] outputs = new double[batchSize][][];
        for(int i = 0; i < batchSize; i++) {
            outputs[i] = forward0(batchData[i]);
        }

        this.cachedOutputs = outputs;
        return outputs;
    }

    @Override
    public double[][][] backwardBatch(double[][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;

        double[][][] gradInputBatch = new double[batchSize][][];
        // 处理每个样本
        for (int b = 0; b < batchSize; b++) {
            double[][] gradInput = backward0(gradOutputBatch[b], b);
            gradInputBatch[b] = gradInput;
        }

        return gradInputBatch;
    }

    @Override
    public void updateParameters() {
        gradOptimizer.applyGradients(wxh, gradWxh);
        gradOptimizer.applyGradients(whh, gradWhh);
        gradOptimizer.applyGradients(bh, gradBh);
        resetGradients();
    }

    private void resetGradients() {
        this.gradWxh = new double[outputDim][inputDim];
        this.gradWhh = new double[outputDim][outputDim];
        this.gradBh = new double[outputDim];
    }


    /**
     * 前向传播（单序列）
     * @param data
     * @return
     */
    private double[][] forward0(double[][] data) {

        int seqLen = data.length;
        double[] hPrev = new double[outputDim]; // h0 = 0,0,0...
        double[][] output = new double[seqLen][outputDim];

        for(int t = 0; t < seqLen; t ++) {
            double[] xT = data[t];

            // 计算：h_t = tanh(Wxh * xT + Whh * h_{t-1} + bh)
            double[] preActivation = new double[outputDim];

            // Wxh * xT
            for(int i = 0; i < outputDim; i ++){
                double sum = 0.0;
                for(int j = 0; j < inputDim; j ++){
                    sum += wxh[i][j] * xT[j];
                }

                for(int j = 0; j < outputDim; j ++){
                    sum += whh[i][j] * hPrev[j];
                }

                // + bh
                sum += bh[i];

                preActivation[i] = sum;

            }

            double[] h_t = activationFunction.activate(preActivation);
            hPrev = h_t;
            output[t] = h_t;
        }

        return output;
    }

    private double[][] backward0(double[][] gradOutput, int index) {
        int seqLen = gradOutput.length;
        double[][] gradInput = new double[seqLen][inputDim];
        double[] dhNext = new double[outputDim]; // 初始为0,0,0...
        for (int t = seqLen - 1; t >= 0; t--) {
            double[] gradHT = gradOutput[t];

            // 合并梯度
            double[] dh = new double[outputDim];
            for (int i = 0; i < outputDim; i++) {
                dh[i] = gradHT[i] + dhNext[i];
            }

            //当前序列的输入
            double[] xT = cachedInputs[index][t];
            //当前序列的输出
            double[] hT = cachedOutputs[index][t];
            //前一序列的输出
            double[] hTPrev = t > 0 ? cachedOutputs[index][t - 1] : new double[outputDim];
            // 激活函数导数：比如dtanh
            double[] dActivation = activationFunction.derivative(hT);

            double[] dpre = new double[outputDim];
            for (int i = 0; i < outputDim; i++) {
                dpre[i] = dh[i] * dActivation[i];
            }

            // 计算输入梯度
            for (int j = 0; j < inputDim; j++) {
                double sum = 0.0;
                for (int i = 0; i < outputDim; i++) {
                    sum += wxh[i][j] * dpre[i];
                }
                gradInput[t][j] = sum;
            }

            // 计算传递给前一个时间步的梯度
            dhNext = new double[outputDim];
            for (int j = 0; j < outputDim; j++) {
                double sum = 0.0;
                for (int i = 0; i < outputDim; i++) {
                    sum += whh[i][j] * dpre[i];
                }
                dhNext[j] = sum;
            }

            // 累加权重梯度
            for (int i = 0; i < outputDim; i++) {
                for (int j = 0; j < inputDim; j++) {
                    // gradWxh += dpre ⊗ x_t
                    gradWxh[i][j] += dpre[i] * xT[j];
                }

                // gradBh += dpre
                gradBh[i] += dpre[i];
            }

            for (int i = 0; i < outputDim; i++) {
                for (int j = 0; j < outputDim; j++) {
                    // gradWhh += dpre ⊗ h_prev
                    gradWhh[i][j] += dpre[i] * hTPrev[j];
                }
            }


        }

        return gradInput;
    }
}