package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.convert.AdaptiveMergeFirstDimsConvert;
import com.oraen.box.otorch.util.DataUtil;
import com.oraen.box.otorch.util.TensorUtil;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

/**
 * Transformer Encoder Block with Pre-LayerNorm (used in LLaMA, GPT, etc.)
 *
 * Forward:
 *   x → LN1 → SelfAttention → +x → LN2 → FFN → +x → output
 *
 * Input/Output shape: [B, T, D]
 */
@Getter
@Setter
public class TransformerEncoderBlock implements Layer<double[][], double[][]>, Learnable {

    private final LayerNorm3D ln1;
    private final MultiHeadSelfAttention selfAttention;
    private final LayerNorm3D ln2;
    private final FeedForwardLayer ffn;

    private final AdaptiveMergeFirstDimsConvert adaptiveMergeFirstDimsConvert = new AdaptiveMergeFirstDimsConvert();

    public TransformerEncoderBlock(MultiHeadSelfAttention selfAttention, LayerNorm3D ln1, FeedForwardLayer ffn, LayerNorm3D ln2) {
        this.selfAttention = selfAttention;
        this.ln1 = ln1;
        this.ffn = ffn;
        this.ln2 = ln2;
    }

    // ================== Forward Pass ==================
    @Override
    public double[][][] forwardBatch(double[][][] data) {

        // ================= LN1 + Self-Attention =================
        double[][][] ln1Out = ln1.forwardBatch(data);
        double[][][] attnOut = selfAttention.forwardBatch(ln1Out);

        // ================= Residual 1 =================
        double[][][] add1 = TensorUtil.add(data, attnOut);

        // ================= LN2 =================
        double[][][] ln2Out = ln2.forwardBatch(add1);

        // ================= FFN =================
        double[][] ln2Flat = adaptiveMergeFirstDimsConvert.forwardBatch(ln2Out);
        double[][] ffnFlatOut = ffn.forwardBatch(ln2Flat);
        double[][][] ffnOut = adaptiveMergeFirstDimsConvert.backwardBatch(ffnFlatOut);

        // ================= Residual 2 =================
        return TensorUtil.add(add1, ffnOut);
    }


    @Override
    public double[][][] backwardBatch(double[][][] gradOutput) {

        // ================= FFN backward =================
        double[][] gradFfnFlat = adaptiveMergeFirstDimsConvert.forwardBatch(gradOutput);
        double[][] gradLn2Flat = ffn.backwardBatch(gradFfnFlat);
        double[][][] gradLn2Out = adaptiveMergeFirstDimsConvert.backwardBatch(gradLn2Flat);

        // ================= LN2 backward =================
        double[][][] gradAdd1FromLn2 = ln2.backwardBatch(gradLn2Out);
        double[][][] gradAdd1 = TensorUtil.add(gradOutput, gradAdd1FromLn2);

        // ================= Self-Attention backward =================
        double[][][] gradLn1Out = selfAttention.backwardBatch(gradAdd1);

        // ================= LN1 backward =================
        double[][][] gradXFromLn1 = ln1.backwardBatch(gradLn1Out);

        // ================= Final gradient =================
        return TensorUtil.add(gradAdd1, gradXFromLn1);
    }

    // ================== Parameter Update ==================
    @Override
    public void updateParameters() {
        selfAttention.updateParameters();
        ffn.updateParameters();
        ln1.updateParameters();
        ln2.updateParameters();
    }

}