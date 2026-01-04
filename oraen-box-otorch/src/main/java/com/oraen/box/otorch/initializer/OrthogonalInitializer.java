package com.oraen.box.otorch.initializer;

import com.oraen.box.otorch.ParamInitializer;

import java.util.Random;

/**
 * 正交初始化器（简化实现）
 * 适用于深度网络，缓解梯度消失/爆炸
 */
public class OrthogonalInitializer implements ParamInitializer {
    private final Random random;
    private final double gain;
    
    public OrthogonalInitializer() {
        this(new Random(), 1.0);
    }
    
    public OrthogonalInitializer(double gain) {
        this(new Random(), gain);
    }
    
    public OrthogonalInitializer(Random random, double gain) {
        this.random = random;
        this.gain = gain;
    }
    
    @Override
    public void initializeWeights(double[][] weights) {
        int fanOut = weights.length;
        int fanIn = weights[0].length;
        
        // 简化的正交初始化
        // 实际的正交初始化需要QR分解或SVD，这里用近似方法
        if (fanOut <= fanIn) {
            initializeSmallOrthogonal(weights, fanOut, fanIn);
        } else {
            initializeLargeOrthogonal(weights, fanOut, fanIn);
        }
    }
    
    private void initializeSmallOrthogonal(double[][] weights, int fanOut, int fanIn) {
        // 当 fanOut <= fanIn 时
        double scale = gain / Math.sqrt(fanIn);
        
        for (int i = 0; i < fanOut; i++) {
            for (int j = 0; j < fanIn; j++) {
                weights[i][j] = random.nextGaussian() * scale;
            }
        }
        
        // 简化的正交化过程（Gram-Schmidt）
        for (int i = 0; i < fanOut; i++) {
            // 归一化当前行
            double norm = 0.0;
            for (int j = 0; j < fanIn; j++) {
                norm += weights[i][j] * weights[i][j];
            }
            norm = Math.sqrt(norm);
            
            if (norm > 1e-10) {
                for (int j = 0; j < fanIn; j++) {
                    weights[i][j] /= norm;
                }
            }
            
            // 正交化后续行
            for (int k = i + 1; k < fanOut; k++) {
                double dot = 0.0;
                for (int j = 0; j < fanIn; j++) {
                    dot += weights[i][j] * weights[k][j];
                }
                
                for (int j = 0; j < fanIn; j++) {
                    weights[k][j] -= dot * weights[i][j];
                }
            }
        }
    }
    
    private void initializeLargeOrthogonal(double[][] weights, int fanOut, int fanIn) {
        // 当 fanOut > fanIn 时，转置处理
        double[][] transposed = new double[fanIn][fanOut];
        initializeSmallOrthogonal(transposed, fanIn, fanOut);
        
        // 转置回来
        for (int i = 0; i < fanOut; i++) {
            for (int j = 0; j < fanIn; j++) {
                weights[i][j] = transposed[j][i] * gain;
            }
        }
    }
    
    @Override
    public void initializeBiases(double[] biases) {
        // 正交初始化通常将偏置设为0
        for (int i = 0; i < biases.length; i++) {
            biases[i] = 0.0;
        }
    }
    
    // 常用配置
    public static OrthogonalInitializer forRNN() {
        return new OrthogonalInitializer(1.0);  // RNN常用
    }
    
    public static OrthogonalInitializer forLSTM() {
        return new OrthogonalInitializer(1.0);
    }
}