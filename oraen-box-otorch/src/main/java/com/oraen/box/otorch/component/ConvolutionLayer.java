package com.oraen.box.otorch.component;

import com.oraen.box.otorch.*;
import lombok.Data;

/**
 * 卷积层
 * 输入形状: [batch, inChannels, height, width]
 * 输出形状: [batch, outChannels, outHeight, outWidth]
 */
@Data
public class ConvolutionLayer implements Layer<double[][][], double[][][]>, Learnable {
    
    // 卷积核参数
    // 形状: [outChannels, inChannels, kernelHeight, kernelWidth]
    private final double[][][][] kernels;
    
    // 偏置参数
    // 形状: [outChannels]
    private final double[] biases;
    
    // 维度信息
    private final int inChannels;
    private final int outChannels;
    private final int kernelHeight;
    private final int kernelWidth;
    
    // 梯度优化器
    private GradOptimizer gradOptimizer;
    
    // 前向传播缓存
    // 缓存输入，用于反向传播
    private double[][][][] cachedInput;
    
    // 梯度累加
    private double[][][][] gradKernelsSum;
    private double[] gradBiasesSum;
    
    // 参数初始化器
    private final ParamInitializer paramInitializer;
    
    public ConvolutionLayer(int inChannels, int outChannels, int kernelHeight, int kernelWidth, ParamInitializer paramInitializer, GradOptimizer gradOptimizer) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.paramInitializer = paramInitializer;
        this.gradOptimizer = gradOptimizer;
        
        // 初始化卷积核
        this.kernels = new double[outChannels][inChannels][kernelHeight][kernelWidth];
        this.biases = new double[outChannels];
        
//        paramInitializer.initializeWeights(kernels);
        paramInitializer.initializeBiases(biases);
        
        resetGradients();
    }
    
    @Override
    public double[][][][] forwardBatch(double[][][][] inputBatch) {
        int batchSize = inputBatch.length;
        int inHeight = inputBatch[0][0].length;
        int inWidth = inputBatch[0][0][0].length;
        
        // 计算输出尺寸（无padding，stride=1）
        int outHeight = inHeight - kernelHeight + 1;
        int outWidth = inWidth - kernelWidth + 1;
        
        // 初始化输出
        double[][][][] outputBatch = new double[batchSize][outChannels][outHeight][outWidth];
        
        // 缓存输入
        this.cachedInput = inputBatch;
        
        // 对每个样本进行卷积
        for (int b = 0; b < batchSize; b++) {
            double[][][] input = inputBatch[b];
            
            // 对每个输出通道
            for (int oc = 0; oc < outChannels; oc++) {
                
                // 对输出空间的每个位置
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        
                        double sum = biases[oc];
                        
                        // 对每个输入通道
                        for (int ic = 0; ic < inChannels; ic++) {
                            // 对卷积核的每个位置
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int ih = oh + kh;
                                    int iw = ow + kw;
                                    sum += input[ic][ih][iw] * kernels[oc][ic][kh][kw];
                                }
                            }
                        }
                        
                        outputBatch[b][oc][oh][ow] = sum;
                    }
                }
            }
        }
        
        return outputBatch;
    }
    
    @Override
    public double[][][][] backwardBatch(double[][][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        int outHeight = gradOutputBatch[0][0].length;
        int outWidth = gradOutputBatch[0][0][0].length;
        int inHeight = cachedInput[0][0].length;
        int inWidth = cachedInput[0][0][0].length;
        
        // 初始化输入梯度
        double[][][][] gradInputBatch = new double[batchSize][inChannels][inHeight][inWidth];
        
        // 重置梯度累加器
        resetGradients();
        
        // 对每个样本
        for (int b = 0; b < batchSize; b++) {
            double[][][] input = cachedInput[b];
            double[][][] gradOutput = gradOutputBatch[b];
            
            // 对每个输出通道
            for (int oc = 0; oc < outChannels; oc++) {
                
                // 对输出空间的每个位置
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        double grad = gradOutput[oc][oh][ow];
                        
                        // 累加偏置梯度
                        gradBiasesSum[oc] += grad;
                        
                        // 对每个输入通道
                        for (int ic = 0; ic < inChannels; ic++) {
                            
                            // 对卷积核的每个位置
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int ih = oh + kh;
                                    int iw = ow + kw;
                                    
                                    // 累加卷积核梯度
                                    gradKernelsSum[oc][ic][kh][kw] += grad * input[ic][ih][iw];
                                    
                                    // 计算输入梯度（全卷积操作）
                                    gradInputBatch[b][ic][ih][iw] += grad * kernels[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return gradInputBatch;
    }
    
    @Override
    public void updateParameters() {
//        // 应用梯度更新
//        GradientsMsg gradientsMsg = new GradientsMsg(gradKernelsSum, gradBiasesSum);
//        gradOptimizer.applyGradients(kernels, biases, gradientsMsg);
//
//        // 重置梯度
//        resetGradients();
    }
    
    private void resetGradients() {
        this.gradKernelsSum = new double[outChannels][inChannels][kernelHeight][kernelWidth];
        this.gradBiasesSum = new double[outChannels];
    }
    
    // 获取输出尺寸（用于构建网络）
    public int[] getOutputShape(int[] inputShape) {
        int outHeight = inputShape[1] - kernelHeight + 1;
        int outWidth = inputShape[2] - kernelWidth + 1;
        return new int[]{outChannels, outHeight, outWidth};
    }
}