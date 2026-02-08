package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.convert.AdaptiveMergeFirstDimsConvert;
import com.oraen.box.otorch.util.TensorUtil;
import lombok.Getter;
import lombok.Setter;

/**
 * Transformer Decoder Block (Decoder-only, GPT/LLaMA style)
 *
 * Forward:
 *   x → LN1 → MaskedSelfAttention → +x → LN2 → FFN → +x → output
 *
 * Shape: [B, T, D]
 */
@Getter
@Setter
public class TransformerDecoderOnlyBlock implements Layer<double[][], double[][]>, Learnable {

    private final LayerNorm3D ln1;
    private final MultiHeadSelfAttention maskedSelfAttention;
    private final LayerNorm3D ln2;
    private final FeedForwardLayer ffn;

    private final AdaptiveMergeFirstDimsConvert adaptiveMergeFirstDimsConvert =
            new AdaptiveMergeFirstDimsConvert();

    public TransformerDecoderOnlyBlock(
            MultiHeadSelfAttention maskedSelfAttention,
            LayerNorm3D ln1,
            FeedForwardLayer ffn,
            LayerNorm3D ln2) {

        this.maskedSelfAttention = maskedSelfAttention;
        this.ln1 = ln1;
        this.ffn = ffn;
        this.ln2 = ln2;
    }

    // ================== Forward ==================

    @Override
    public double[][][] forwardBatch(double[][][] data) {

        // ===== LN1 + Masked Self Attention =====
        double[][][] ln1Out = ln1.forwardBatch(data);
        double[][][] attnOut = maskedSelfAttention.forwardBatch(ln1Out);

        // ===== Residual 1 =====
        double[][][] add1 = TensorUtil.add(data, attnOut);

        // ===== LN2 =====
        double[][][] ln2Out = ln2.forwardBatch(add1);

        // ===== FFN =====
        double[][] ln2Flat = adaptiveMergeFirstDimsConvert.forwardBatch(ln2Out);
        double[][] ffnFlatOut = ffn.forwardBatch(ln2Flat);
        double[][][] ffnOut = adaptiveMergeFirstDimsConvert.backwardBatch(ffnFlatOut);

        // ===== Residual 2 =====
        return TensorUtil.add(add1, ffnOut);
    }


    // ================== Backward ==================

    @Override
    public double[][][] backwardBatch(double[][][] gradOutput) {

        // ===== FFN backward =====
        double[][] gradFfnFlat =
                adaptiveMergeFirstDimsConvert.forwardBatch(gradOutput);

        double[][] gradLn2Flat =
                ffn.backwardBatch(gradFfnFlat);

        double[][][] gradLn2Out =
                adaptiveMergeFirstDimsConvert.backwardBatch(gradLn2Flat);

        // ===== LN2 backward =====
        double[][][] gradAdd1FromLn2 =
                ln2.backwardBatch(gradLn2Out);

        double[][][] gradAdd1 =
                TensorUtil.add(gradOutput, gradAdd1FromLn2);

        // ===== Masked Self Attention backward =====
        double[][][] gradLn1Out =
                maskedSelfAttention.backwardBatch(gradAdd1);

        // ===== LN1 backward =====
        double[][][] gradXFromLn1 =
                ln1.backwardBatch(gradLn1Out);

        // ===== Final gradient =====
        return TensorUtil.add(gradAdd1, gradXFromLn1);
    }


    // ================== Update ==================

    @Override
    public void updateParameters() {
        maskedSelfAttention.updateParameters();
        ffn.updateParameters();
        ln1.updateParameters();
        ln2.updateParameters();
    }
}
