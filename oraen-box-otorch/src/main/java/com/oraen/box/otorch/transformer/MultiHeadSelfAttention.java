package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.optimizer.AdamWOptimizer;
import com.oraen.box.otorch.util.DataUtil;
import com.oraen.box.otorch.util.TensorUtil;
import lombok.Getter;
import lombok.Setter;


/**
 * 多头自注意力（Multi-Head Self Attention）with RoPE
 * 其中:
 * B = batch size
 * T = 序列长度
 * D = embedDim
 * H = numHeads
 * Dh = headDim = D / H
 */
@Getter
public class MultiHeadSelfAttention implements AttentionLayer, Learnable {

    private final int embedDim;      // 总嵌入维度 D
    private final int numHeads;      // 头数 H
    private final int headDim;       // 每个头的维度 Dh = D / H

    private final boolean useMask;
    private final SingleHeadSelfAttention.PositionalEncodingType positionalEncodingType;

    // 多个单头注意力层
    private final SingleHeadSelfAttention[] heads;

    // 输出投影矩阵 Wo: [D, D] - 将拼接后的多头输出映射回原始维度
    private final double[][] Wo;
    private final double[][] gradWo;

    @Setter
    private GradOptimizer woOptimizer;

    // 缓存用于反向传播
    private double[][][] input;           // [B, T, D]
    private double[][][] concatenatedHeadsOut; // [B, T, D] - 拼接后的多头输出

    public MultiHeadSelfAttention(int embedDim, int numHeads, GradOptimizer woOptimizer, ParamInitializer paramInitializer,
                                  SingleHeadSelfAttention.PositionalEncodingType positionalEncodingType, boolean useMask) {

        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.useMask = useMask;
        this.positionalEncodingType = positionalEncodingType;
        this.woOptimizer = woOptimizer;

        // 初始化多个单头注意力层
        this.heads = new SingleHeadSelfAttention[numHeads];
        for (int i = 0; i < numHeads; i++) {
            heads[i] = new SingleHeadSelfAttention(embedDim, headDim,
                    new AdamWOptimizer(this.headDim, embedDim),
                    new AdamWOptimizer(this.headDim, embedDim),
                    new AdamWOptimizer(this.headDim, embedDim),
                    paramInitializer,
                    positionalEncodingType,
                    useMask
            );
        }

        // 初始化输出投影矩阵 Wo: [D, D]
        this.Wo = new double[embedDim][embedDim];
        this.gradWo = new double[embedDim][embedDim];
        paramInitializer.initializeWeights(Wo);
    }

    public MultiHeadSelfAttention(SingleHeadSelfAttention[] singleHeadSelfAttentions, GradOptimizer woOptimizer, ParamInitializer paramInitializer,
                                  SingleHeadSelfAttention.PositionalEncodingType positionalEncodingType, boolean useMask) {

        this.heads = new SingleHeadSelfAttention[singleHeadSelfAttentions.length];
        this.embedDim = singleHeadSelfAttentions[0].getInputDim();
        this.numHeads = singleHeadSelfAttentions.length;
        if(embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.headDim = embedDim / numHeads;
        for(int i = 0; i < singleHeadSelfAttentions.length; i++) {
            if(this.headDim != singleHeadSelfAttentions[i].getOutputDim()) {
                throw new IllegalArgumentException("headDim must be the same for all heads");
            }

            if(this.embedDim != singleHeadSelfAttentions[i].getInputDim()) {
                throw new IllegalArgumentException("embedDim must be the same for all heads");
            }

            this.heads[i] = singleHeadSelfAttentions[i];
        }

        this.useMask = useMask;
        this.positionalEncodingType = positionalEncodingType;
        this.woOptimizer = woOptimizer;

        // 初始化输出投影矩阵 Wo: [D, D]
        this.Wo = new double[embedDim][embedDim];
        this.gradWo = new double[embedDim][embedDim];
        paramInitializer.initializeWeights(Wo);
    }

    @Override
    public double[][][] forwardBatch(double[][][] data) {
        this.input = data;
        int batchSize = data.length;
        int seqLen = data[0].length;

        // 存储每个头的输出
        double[][][][] headOutputs = new double[numHeads][][][];

        // 前向传播每个头
        for (int h = 0; h < numHeads; h++) {
            double[][][] headOut = heads[h].forwardBatch(data);
            headOutputs[h] = headOut;
        }

        // 拼接所有头的输出: [B, T, H * Dh] = [B, T, D]
        this.concatenatedHeadsOut = combine(headOutputs);

        // 应用输出投影: output = concatenatedHeads @ Wo^T
        double[][][] finalOutput = new double[batchSize][seqLen][embedDim];
        for (int b = 0; b < batchSize; b++) {
            finalOutput[b] = TensorUtil.project(concatenatedHeadsOut[b], Wo);
        }

        return finalOutput;
    }

    @Override
    public double[][][] backwardBatch(double[][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        int seqLen = gradOutputBatch[0].length;
        double[][][] gradInput = new double[batchSize][seqLen][embedDim];

        // Step 1: 计算 Wo 的梯度和拼接输出的梯度
        double[][][] gradConcatenated = new double[batchSize][seqLen][embedDim];
        for (int b = 0; b < batchSize; b++) {
            // gradConcatenated[b] = gradOutputBatch[b] @ Wo
            gradConcatenated[b] = TensorUtil.matmul(gradOutputBatch[b], Wo);

            // 累加 Wo 的梯度: gradWo += gradOutputBatch[b]^T @ concatenatedHeads[b]
            accumulateGrad(gradWo, gradOutputBatch[b], concatenatedHeadsOut[b]);
        }

        // Step 2: 分割梯度到各个头
        double[][][][] headGradOutputs = new double[numHeads][batchSize][seqLen][headDim];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                int offset = 0;
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        headGradOutputs[h][b][t][d] = gradConcatenated[b][t][offset + d];
                    }
                    offset += headDim;
                }
            }
        }

        // Step 3: 反向传播每个头
        for (int h = 0; h < numHeads; h++) {
            double[][][] headGradInput = heads[h].backwardBatch(headGradOutputs[h]);
            // 累加所有头的输入梯度
            for (int b = 0; b < batchSize; b++) {
                for (int t = 0; t < seqLen; t++) {
                    for (int d = 0; d < embedDim; d++) {
                        gradInput[b][t][d] += headGradInput[b][t][d];
                    }
                }
            }
        }

        return gradInput;
    }

    @Override
    public void updateParameters() {
        // 更新所有头的参数
        for (SingleHeadSelfAttention head : heads) {
            head.updateParameters();
        }
        // 更新输出投影矩阵
        woOptimizer.applyGradients(Wo, gradWo);
        clearGrad();
    }

    private void clearGrad() {
        // 清除 Wo 梯度
        for (int i = 0; i < embedDim; i++) {
            for (int j = 0; j < embedDim; j++) {
                gradWo[i][j] = 0.0;
            }
        }
    }

    private void accumulateGrad(double[][] gradW, double[][] gradAct, double[][] input) {
        // gradW += gradAct^T @ input
        int outDim = gradW.length;
        int inDim = gradW[0].length;
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                double sum = 0.0;
                for (int t = 0; t < gradAct.length; t++) {
                    sum += gradAct[t][i] * input[t][j];
                }
                gradW[i][j] += sum;
            }
        }
    }

    private double[][][] combine(double[][][][] headOutputs){
        int numHeads = headOutputs.length;
        int batchSize = headOutputs[0].length;
        double[][][] re = new double[batchSize][][];
        for(int b = 0; b < batchSize; b++){
            int seqLen = headOutputs[0][b].length;
            double[][] outItem = new double[seqLen][embedDim];
            for(int h = 0; h < numHeads; h++){
                int offset = h * headDim;
                DataUtil.paintData(headOutputs[h][b], outItem, 0, offset);
            }
            re[b] = outItem;
        }
        return re;
    }


}