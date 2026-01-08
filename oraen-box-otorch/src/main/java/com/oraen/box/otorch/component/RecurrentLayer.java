package com.oraen.box.otorch.component;

import com.oraen.box.otorch.*;
import lombok.Data;

import java.util.Arrays;
import java.util.List;

/**
 * RNN层（填充方案）
 * 输入形状：[batch_size, seq_len, input_dim]
 * 输出形状：[batch_size, seq_len, hidden_dim]
 * 支持可变长度（通过seqLen参数）
 */
@Data
public class RecurrentLayer implements Layer<double[][], double[][]>, Learnable {

    // 输入到隐藏层权重，将输入转换到隐藏空间
    private final double[][] Wxh;
    // 隐藏层递归权重，记忆跨时间步信息
    private final double[][] Whh;
    // 隐藏层偏置
    private final double[] bh;

    // 输入特征维度
    private final int inputDim;
    // 隐藏状态维度，决定记忆容量
    private final int hiddenDim;

    // 梯度优化器
    // 权重更新策略
    private GradOptimizer gradOptimizer;

    // 前向传播缓存（用于反向传播）
    // 各时间步输入，计算Wxh梯度
    private double[][][] cachedInputs;
    // 各时间步隐藏状态，计算Whh梯度
    private double[][][] cachedHiddenStates;
    // 激活前线性变换结果，计算激活函数导数
    private double[][][] cachedPreActivations;
    // 实际序列长度，处理可变长度序列
    private int[] cachedSeqLens;

    // 梯度累加
    // Wxh梯度累加和
    private double[][] gradWxhSum;
    // Whh梯度累加和
    private double[][] gradWhhSum;
    // bh梯度累加和
    private double[] gradBhSum;



    public RecurrentLayer(int inputDim, int hiddenDim, ParamInitializer paramInitializer, GradOptimizer gradOptimizer) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.gradOptimizer = gradOptimizer;

        // 初始化权重
        this.Wxh = new double[hiddenDim][inputDim];
        this.Whh = new double[hiddenDim][hiddenDim];
        this.bh = new double[hiddenDim];

        paramInitializer.initializeWeights(Wxh);
        paramInitializer.initializeWeights(Whh);
        paramInitializer.initializeBiases(bh);

        resetGradients();
    }

    @Override
    public double[][][] forwardBatch(double[][][] batchData) {
        int batchSize = batchData.length;
        int maxSeqLen = batchData[0].length; // 最大序列长度（填充后）

        // 确定实际序列长度（简化：假设输入不为0的部分）
        this.cachedSeqLens = new int[batchSize];
        for (int b = 0; b < batchSize; b++) {
            cachedSeqLens[b] = maxSeqLen; // 简化，实际需要检测填充
        }

        // 初始化缓存
        this.cachedInputs = batchData;
        this.cachedHiddenStates = new double[batchSize][maxSeqLen + 1][hiddenDim];
        this.cachedPreActivations = new double[batchSize][maxSeqLen][hiddenDim];

        double[][][] batchOutput = new double[batchSize][maxSeqLen][hiddenDim];

        // 处理每个样本
        for (int b = 0; b < batchSize; b++) {
            int seqLen = cachedSeqLens[b];
            double[] hPrev = new double[hiddenDim]; // h0 = 0

            for (int t = 0; t < seqLen; t++) {
                double[] x_t = batchData[b][t];

                // 计算：h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
                double[] preActivation = new double[hiddenDim];

                // Wxh * x_t
                for (int i = 0; i < hiddenDim; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < inputDim; j++) {
                        sum += Wxh[i][j] * x_t[j];
                    }
                    preActivation[i] = sum;
                }

                // + Whh * h_{t-1}
                for (int i = 0; i < hiddenDim; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < hiddenDim; j++) {
                        sum += Whh[i][j] * hPrev[j];
                    }
                    preActivation[i] += sum;
                }

                // + bh
                for (int i = 0; i < hiddenDim; i++) {
                    preActivation[i] += bh[i];
                }

                // 缓存
                cachedPreActivations[b][t] = preActivation;

                // tanh激活
                double[] h_t = new double[hiddenDim];
                for (int i = 0; i < hiddenDim; i++) {
                    h_t[i] = Math.tanh(preActivation[i]);
                }

                // 保存
                cachedHiddenStates[b][t] = hPrev;      // 保存h_{t-1}
                batchOutput[b][t] = h_t;
                hPrev = h_t;
            }
            cachedHiddenStates[b][seqLen] = hPrev; // 保存最后一个h_t
        }

        return batchOutput;
    }

    @Override
    public double[][][] backwardBatch(double[][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        int maxSeqLen = gradOutputBatch[0].length;

        double[][][] gradInputBatch = new double[batchSize][maxSeqLen][inputDim];

        // 重置梯度累加
        resetGradients();

        // 处理每个样本
        for (int b = 0; b < batchSize; b++) {
            int seqLen = cachedSeqLens[b];
            double[] dh_next = new double[hiddenDim]; // 初始为0

            // 反向遍历时间步
            for (int t = seqLen - 1; t >= 0; t--) {
                double[] grad_h_t = gradOutputBatch[b][t];

                // 合并梯度
                double[] dh = new double[hiddenDim];
                for (int i = 0; i < hiddenDim; i++) {
                    dh[i] = grad_h_t[i] + dh_next[i];
                }

                // 获取缓存值
                double[] preActivation = cachedPreActivations[b][t];
                double[] x_t = cachedInputs[b][t];
                double[] h_prev = cachedHiddenStates[b][t];
                double[] h_t = cachedHiddenStates[b][t + 1];

                // tanh导数：dtanh = 1 - tanh²
                double[] dtanh = new double[hiddenDim];
                for (int i = 0; i < hiddenDim; i++) {
                    dtanh[i] = 1 - h_t[i] * h_t[i];
                }

                // dpre = dh ⊙ dtanh
                double[] dpre = new double[hiddenDim];
                for (int i = 0; i < hiddenDim; i++) {
                    dpre[i] = dh[i] * dtanh[i];
                }

                // 计算输入梯度
                for (int j = 0; j < inputDim; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < hiddenDim; i++) {
                        sum += Wxh[i][j] * dpre[i];
                    }
                    gradInputBatch[b][t][j] = sum;
                }

                // 累加权重梯度
                // gradWxh += dpre ⊗ x_t
                for (int i = 0; i < hiddenDim; i++) {
                    for (int j = 0; j < inputDim; j++) {
                        gradWxhSum[i][j] += dpre[i] * x_t[j];
                    }
                }

                // gradWhh += dpre ⊗ h_prev
                for (int i = 0; i < hiddenDim; i++) {
                    for (int j = 0; j < hiddenDim; j++) {
                        gradWhhSum[i][j] += dpre[i] * h_prev[j];
                    }
                }

                // gradBh += dpre
                for (int i = 0; i < hiddenDim; i++) {
                    gradBhSum[i] += dpre[i];
                }

                // 计算传递给前一个时间步的梯度
                dh_next = new double[hiddenDim];
                for (int j = 0; j < hiddenDim; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < hiddenDim; i++) {
                        sum += Whh[i][j] * dpre[i];
                    }
                    dh_next[j] = sum;
                }
            }
        }

        return gradInputBatch;
    }

    @Override
    public void updateParameters() {
        // 更新Wxh和bh
        GradientsMsg gradientsMsg1 = new GradientsMsg(gradWxhSum, gradBhSum);
        gradOptimizer.applyGradients(Wxh, bh, gradientsMsg1);

        // 更新Whh（偏置为0）
        GradientsMsg gradientsMsg2 = new GradientsMsg(gradWhhSum, new double[hiddenDim]);
        gradOptimizer.applyGradients(Whh, new double[hiddenDim], gradientsMsg2);

        resetGradients();
    }

    private void resetGradients() {
        this.gradWxhSum = new double[hiddenDim][inputDim];
        this.gradWhhSum = new double[hiddenDim][hiddenDim];
        this.gradBhSum = new double[hiddenDim];
    }

    /**
     * 工具方法：填充序列到统一长度
     */
    public static double[][][] padSequences(List<double[]>[] sequences, int maxLen, int inputDim) {
        int batchSize = sequences.length;
        double[][][] padded = new double[batchSize][maxLen][inputDim];

        for (int b = 0; b < batchSize; b++) {
            List<double[]> seq = sequences[b];
            int seqLen = seq.size();

            // 复制数据
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(seq.get(t), 0, padded[b][t], 0, inputDim);
            }

            // 填充部分保持为0
            for (int t = seqLen; t < maxLen; t++) {
                Arrays.fill(padded[b][t], 0.0);
            }
        }

        return padded;
    }

    /**
     * 工具方法：创建序列长度数组
     */
    public static int[] getSequenceLengths(List<double[]>[] sequences) {
        int[] lengths = new int[sequences.length];
        for (int i = 0; i < sequences.length; i++) {
            lengths[i] = sequences[i].size();
        }
        return lengths;
    }
}