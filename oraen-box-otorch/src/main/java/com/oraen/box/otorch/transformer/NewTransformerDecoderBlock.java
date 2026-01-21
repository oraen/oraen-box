package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import lombok.Getter;
import lombok.Setter;

/**
 * Transformer Decoder Block for Decoder-only models (GPT, LLaMA, etc.)
 *
 * Architecture (Pre-LN):
 *   x → LN1 → MaskedSelfAttention → +x → LN2 → FFN → +x → output
 *
 * Input/Output shape: [B, T, D]
 */
@Getter
@Setter
public class NewTransformerDecoderBlock implements Layer<double[][], double[][]>, Learnable {

    private final MultiHeadSelfAttention selfAttention; // must support causal masking
    private final LayerNorm ln1;
    private final FeedForwardLayer ffn;
    private final LayerNorm ln2;



    public NewTransformerDecoderBlock(
            MultiHeadSelfAttention selfAttention,
            LayerNorm ln1,
            FeedForwardLayer ffn,
            LayerNorm ln2) {
        this.selfAttention = selfAttention;
        this.ln1 = ln1;
        this.ffn = ffn;
        this.ln2 = ln2;
    }

    // ================== Forward Pass ==================
    @Override
    public double[][][] forwardBatch(double[][][] data) {
        int B = data.length;
        int T = data[0].length;
        int D = data[0][0].length;

        // ---- Pre-LN: LayerNorm → Masked Self-Attention → Residual ----
        double[][][] ln1Out = reshape3D(ln1.forwardBatch(flatten(data)), B, T, D);
        double[][][] attnOut = selfAttention.forwardBatch(ln1Out); // assumes isCausal=true

        double[][][] add1 = new double[B][T][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < D; d++) {
                    add1[b][t][d] = data[b][t][d] + attnOut[b][t][d];
                }
            }
        }

        // ---- Pre-LN: LayerNorm → FFN → Residual ----
        double[][][] ln2Out = reshape3D(ln2.forwardBatch(flatten(add1)), B, T, D);
        double[][][] ffnOut = reshape3D(ffn.forwardBatch(flatten(ln2Out)), B, T, D);

        double[][][] output = new double[B][T][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < D; d++) {
                    output[b][t][d] = add1[b][t][d] + ffnOut[b][t][d];
                }
            }
        }

        return output;
    }

    // ================== Backward Pass ==================
    @Override
    public double[][][] backwardBatch(double[][][] gradOutput) {
        int B = gradOutput.length;
        int T = gradOutput[0].length;
        int D = gradOutput[0][0].length;

        // ---- Step 1: FFN residual backward ----
        double[][][] gradAdd1 = copy(gradOutput);         // to add1 (residual)
        double[][][] gradFfnOut = copy(gradOutput);       // to ffnOut

        // FFN backward
        double[][] gradFlatFfn = ffn.backwardBatch(flatten(gradFfnOut));
        double[][][] gradLn2Out = reshape3D(gradFlatFfn, B, T, D);

        // LN2 backward
        double[][] gradFlatAdd1FromLN2 = ln2.backwardBatch(flatten(gradLn2Out));
        double[][][] gradAdd1FromLN2 = reshape3D(gradFlatAdd1FromLN2, B, T, D);

        // Accumulate gradient into add1
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < D; d++) {
                    gradAdd1[b][t][d] += gradAdd1FromLN2[b][t][d];
                }
            }
        }

        // ---- Step 2: Self-Attention residual backward ----
        double[][][] gradX = copy(gradAdd1);              // to original input (residual)
        double[][][] gradAttnOut = copy(gradAdd1);        // to attention output

        // Self-Attention backward
        double[][][] gradLn1Out = selfAttention.backwardBatch(gradAttnOut);

        // LN1 backward
        double[][] gradFlatXFromLN1 = ln1.backwardBatch(flatten(gradLn1Out));
        double[][][] gradXFromLN1 = reshape3D(gradFlatXFromLN1, B, T, D);

        // Accumulate gradient from LN1 path
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < D; d++) {
                    gradX[b][t][d] += gradXFromLN1[b][t][d];
                }
            }
        }

        return gradX;
    }

    // ================== Parameter Update ==================
    @Override
    public void updateParameters() {
        selfAttention.updateParameters();
        ffn.updateParameters();
        ln1.updateParameters();
        ln2.updateParameters();
    }

    // ================== Utility Methods ==================
    private static double[][] flatten(double[][][] x) {
        int B = x.length;
        int T = x[0].length;
        int D = x[0][0].length;
        double[][] out = new double[B * T][D];
        int idx = 0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                System.arraycopy(x[b][t], 0, out[idx++], 0, D);
            }
        }
        return out;
    }

    private static double[][][] reshape3D(double[][] x, int B, int T, int D) {
        double[][][] out = new double[B][T][D];
        int idx = 0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                out[b][t] = x[idx++];
            }
        }
        return out;
    }

    private static double[][][] copy(double[][][] src) {
        int B = src.length;
        int T = src[0].length;
        int D = src[0][0].length;
        double[][][] dst = new double[B][T][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                System.arraycopy(src[b][t], 0, dst[b][t], 0, D);
            }
        }
        return dst;
    }
}


