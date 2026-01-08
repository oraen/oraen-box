package com.oraen.box.otorch.component;

import com.oraen.box.otorch.Layer;

/**
 * 最大池化层
 * 输入形状: [batch, channels, height, width]
 * 输出形状: [batch, channels, outHeight, outWidth]
 */
public class PoolLayer implements Layer<double[][][], double[][][]> {
    
    // 池化窗口大小
    private final int poolHeight;
    private final int poolWidth;
    
    // 前向传播缓存
    // 记录最大值位置，用于反向传播
    private int[][][][][] maxPositions; // [batch, channel, outH, outW, 2] 记录(h,w)
    
    public PoolLayer(int poolHeight, int poolWidth) {
        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
    }
    
    @Override
    public double[][][][] forwardBatch(double[][][][] inputBatch) {
        int batchSize = inputBatch.length;
        int channels = inputBatch[0].length;
        int inHeight = inputBatch[0][0].length;
        int inWidth = inputBatch[0][0][0].length;
        
        // 计算输出尺寸
        int outHeight = inHeight / poolHeight;
        int outWidth = inWidth / poolWidth;
        
        // 初始化输出和缓存
        double[][][][] outputBatch = new double[batchSize][channels][outHeight][outWidth];
        this.maxPositions = new int[batchSize][channels][outHeight][outWidth][2];
        
        // 对每个样本
        for (int b = 0; b < batchSize; b++) {
            double[][][] input = inputBatch[b];
            
            // 对每个通道
            for (int c = 0; c < channels; c++) {
                
                // 对输出空间的每个位置
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        
                        // 池化窗口的起始位置
                        int startH = oh * poolHeight;
                        int startW = ow * poolWidth;
                        
                        // 寻找窗口内的最大值
                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxH = startH;
                        int maxW = startW;
                        
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = startH + ph;
                                int iw = startW + pw;
                                double val = input[c][ih][iw];
                                
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }
                        
                        // 保存结果和最大值位置
                        outputBatch[b][c][oh][ow] = maxVal;
                        maxPositions[b][c][oh][ow][0] = maxH;
                        maxPositions[b][c][oh][ow][1] = maxW;
                    }
                }
            }
        }
        
        return outputBatch;
    }
    
    @Override
    public double[][][][] backwardBatch(double[][][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        int channels = gradOutputBatch[0].length;
        int outHeight = gradOutputBatch[0][0].length;
        int outWidth = gradOutputBatch[0][0][0].length;
        
        // 获取输入尺寸（从缓存的位置信息推断）
        int inHeight = maxPositions[0][0][0][0][0] + 1; // 简化
        int inWidth = maxPositions[0][0][0][0][1] + 1;
        
        // 初始化输入梯度
        double[][][][] gradInputBatch = new double[batchSize][channels][inHeight][inWidth];
        
        // 对每个样本
        for (int b = 0; b < batchSize; b++) {
            
            // 对每个通道
            for (int c = 0; c < channels; c++) {
                
                // 对输出空间的每个位置
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        
                        // 获取最大值位置
                        int maxH = maxPositions[b][c][oh][ow][0];
                        int maxW = maxPositions[b][c][oh][ow][1];
                        
                        // 梯度只传递给最大值位置（最大池化的特性）
                        gradInputBatch[b][c][maxH][maxW] += gradOutputBatch[b][c][oh][ow];
                    }
                }
            }
        }
        
        return gradInputBatch;
    }
    
    // 获取输出尺寸
    public int[] getOutputShape(int[] inputShape) {
        int outHeight = inputShape[1] / poolHeight;
        int outWidth = inputShape[2] / poolWidth;
        return new int[]{inputShape[0], outHeight, outWidth};
    }
    
    // 平均池化版本（可选）
    public class AvgPoolLayer extends PoolLayer {
        public AvgPoolLayer(int poolHeight, int poolWidth) {
            super(poolHeight, poolWidth);
        }
        
        @Override
        public double[][][][] forwardBatch(double[][][][] inputBatch) {
            int batchSize = inputBatch.length;
            int channels = inputBatch[0].length;
            int inHeight = inputBatch[0][0].length;
            int inWidth = inputBatch[0][0][0].length;
            
            int outHeight = inHeight / poolHeight;
            int outWidth = inWidth / poolWidth;
            
            double[][][][] outputBatch = new double[batchSize][channels][outHeight][outWidth];
            
            for (int b = 0; b < batchSize; b++) {
                double[][][] input = inputBatch[b];
                
                for (int c = 0; c < channels; c++) {
                    for (int oh = 0; oh < outHeight; oh++) {
                        for (int ow = 0; ow < outWidth; ow++) {
                            
                            int startH = oh * poolHeight;
                            int startW = ow * poolWidth;
                            double sum = 0.0;
                            
                            for (int ph = 0; ph < poolHeight; ph++) {
                                for (int pw = 0; pw < poolWidth; pw++) {
                                    sum += input[c][startH + ph][startW + pw];
                                }
                            }
                            
                            outputBatch[b][c][oh][ow] = sum / (poolHeight * poolWidth);
                        }
                    }
                }
            }
            
            return outputBatch;
        }
        
        @Override
        public double[][][][] backwardBatch(double[][][][] gradOutputBatch) {
            int batchSize = gradOutputBatch.length;
            int channels = gradOutputBatch[0].length;
            int outHeight = gradOutputBatch[0][0].length;
            int outWidth = gradOutputBatch[0][0][0].length;
            
            int inHeight = outHeight * poolHeight;
            int inWidth = outWidth * poolWidth;
            
            double[][][][] gradInputBatch = new double[batchSize][channels][inHeight][inWidth];
            double scale = 1.0 / (poolHeight * poolWidth);
            
            for (int b = 0; b < batchSize; b++) {
                for (int c = 0; c < channels; c++) {
                    for (int oh = 0; oh < outHeight; oh++) {
                        for (int ow = 0; ow < outWidth; ow++) {
                            
                            double grad = gradOutputBatch[b][c][oh][ow] * scale;
                            int startH = oh * poolHeight;
                            int startW = ow * poolWidth;
                            
                            for (int ph = 0; ph < poolHeight; ph++) {
                                for (int pw = 0; pw < poolWidth; pw++) {
                                    gradInputBatch[b][c][startH + ph][startW + pw] += grad;
                                }
                            }
                        }
                    }
                }
            }
            
            return gradInputBatch;
        }
    }
}