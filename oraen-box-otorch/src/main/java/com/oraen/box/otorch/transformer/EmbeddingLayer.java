package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.util.DataUtil;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

@Getter
public class EmbeddingLayer implements Layer<int[], double[][]>, Learnable {


    // 词表大小
    private final int vocabSize;
    // embedding 维度
    private final int embedDim;

    // embedding 参数表 [vocabSize][embedDim]
    private final double[][] embeddingTable;

    // embedding 梯度表
    private final double[][] gradEmbedding;

    // forward 缓存：token ids（batch × seqLen）
    private int[][] inputIds;

    @Setter
    private GradOptimizer optimizer;

    public EmbeddingLayer(int vocabSize, int embedDim, GradOptimizer optimizer, ParamInitializer embeddingInitializer) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.optimizer = optimizer;

        this.embeddingTable = new double[vocabSize][embedDim];
        this.gradEmbedding = new double[vocabSize][embedDim];

        // 初始化 embedding
        embeddingInitializer.initializeWeights(this.embeddingTable);
    }

    /**
     * forward: token ids -> embedding vectors
     * input: [batch][seqLen]
     * output: [batch][seqLen][embedDim]
     */
    @Override
    public double[][][] forwardBatch(int[][] data) {
        this.inputIds = data;
        int batchSize = data.length;

        double[][][] output = new double[batchSize][][];

        for (int b = 0; b < batchSize; b++) {
            output[b] = forward0(data[b]);
        }

        return output;
    }

    /**
     * backward: 累加 embedding 梯度
     * gradOutput: [batch][seqLen][embedDim]
     */
    @Override
    public int[][] backwardBatch(double[][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;

        for (int b = 0; b < batchSize; b++) {
            backward(gradOutputBatch[b], inputIds[b]);
        }

        // Embedding已经是顶层了 不向前传梯度
        return null;
    }

    @Override
    public void updateParameters() {
        optimizer.applyGradients(embeddingTable, gradEmbedding);
        clearGrad();
    }

    private void clearGrad() {
        for (int i = 0; i < vocabSize; i++) {
            Arrays.fill(gradEmbedding[i], 0.0);
        }
    }



    private double[][] forward0(int[] data) {
        int seqLen = data.length;

        double[][] output = new double[seqLen][embedDim];
        for (int t = 0; t < seqLen; t++) {
            int tokenId = data[t];
            // embeddingTable[tokenId] 是该 token 的向量表示，把这个向量逐元素复制到输出中
            output[t] = DataUtil.copy(embeddingTable[tokenId]);
        }

        return output;
    }

    private void backward(double[][] gradOutput, int[] inputIds) {
        int seqLen = gradOutput.length;
        for (int t = 0; t < seqLen; t++) {
            int tokenId = inputIds[t];
            for (int i = 0; i < embedDim; i++) {
                gradEmbedding[tokenId][i] += gradOutput[t][i];
            }
        }


    }

}