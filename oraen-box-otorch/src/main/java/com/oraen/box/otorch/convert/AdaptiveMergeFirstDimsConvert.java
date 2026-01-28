package com.oraen.box.otorch.convert;

import com.oraen.box.otorch.Layer;

/* =====================================================
 * Adaptive version: Merge first two dims without fixed shapes
 *
 * 输入(单样本): double[firstDim][seqLen_i][hidden]
 * 其中 seqLen_i 在同一 batch 内可以不同
 *
 * 设计原则：
 * - forwardBatch 记录本次 batch 的结构信息
 * - backwardBatch 必须与最近一次 forwardBatch 成对使用
 * - 不可并发 / 不可重入（和大多数 Layer 一致）
 * ===================================================== */
public class AdaptiveMergeFirstDimsConvert implements Layer<double[][], double[]> {


    /**
     * 本次 batch 的 firstDim（batch size）
     */
    private int firstDim;

    /**
     * 记录每个样本的 secondDim（序列长度）
     */
    private int[] secondDims;


    @Override
    public double[][] forwardBatch(double[][][] data) {
        this.firstDim = data.length;
        this.secondDims = new int[firstDim];

        int totalRows = 0;
        for (int i = 0; i < firstDim; i++) {
            secondDims[i] = data[i].length;
            totalRows += secondDims[i];
        }

        double[][] out = new double[totalRows][];

        int row = 0;
        for (int i = 0; i < firstDim; i++) {
            for (int j = 0; j < secondDims[i]; j++) {
                out[row ++] = data[i][j];
            }
        }
        return out;
    }


    @Override
    public double[][][] backwardBatch(double[][] data) {
        if (secondDims == null) {
            throw new IllegalStateException("backwardBatch called before forwardBatch");
        }


        double[][][] out = new double[firstDim][][];


        int row = 0;
        for (int i = 0; i < firstDim; i++) {
            int seqLen = secondDims[i];
            out[i] = new double[seqLen][];
            for (int j = 0; j < seqLen; j++) {
                out[i][j] = data[row ++];
            }
        }


        return out;
    }
}