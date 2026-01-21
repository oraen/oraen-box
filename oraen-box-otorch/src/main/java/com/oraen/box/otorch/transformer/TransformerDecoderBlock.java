package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.GradOptimizer;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

/**
 * Classic Transformer Decoder Block (with Cross-Attention)
 *
 * Architecture (Pre-LN):
 *   x ──→ LN1 ──→ MaskedSelfAttn ──→ +x ──→ LN2 ──→ CrossAttn (Q=x, K=V=enc) ──→ +x ──→ LN3 ──→ FFN ──→ +x
 *
 * Forward inputs:
 *   - decoderInput: [B, T_dec, D]
 *   - encoderOutput: [B, T_enc, D]
 *
 * Backward outputs:
 *   - grad to decoder input
 *   - grad to encoder output (via output parameter)
 */
@Getter
@Setter
public class TransformerDecoderBlock implements Learnable {

    private final MultiHeadSelfAttention maskedSelfAttn; // isCausal = true
    private final LayerNorm ln1;

    // Cross-Attention weights
    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final double scale;

    private final double[][] Wq_cross;
    private final double[][] Wk_cross;
    private final double[][] Wv_cross;
    private final double[][] Wo_cross;

    private final double[][] gradWq_cross;
    private final double[][] gradWk_cross;
    private final double[][] gradWv_cross;
    private final double[][] gradWo_cross;

    private final LayerNorm ln2;
    private final FeedForwardLayer ffn;
    private final LayerNorm ln3;
    private GradOptimizer optimizer;

    // === Forward cache ===
    private double[][][] decoderInput;
    private double[][][] encoderOutput;

    private double[][][] ln1Out;
    private double[][][] selfAttnOut;
    private double[][][] add1; // after self-attn residual

    // Cross-attention internal states
    private double[][][][] q_cross;
    private double[][][][] k_cross;
    private double[][][][] v_cross;
    private double[][][][] attn_weights_cross; // [B, H, T_dec, T_enc]
    private double[][][] context_concat; // [B, T_dec, D]

    private double[][][] ln2Out;
    private double[][][] crossAttnOut;
    private double[][][] add2; // after cross-attn residual

    private double[][][] ln3Out;
    private double[][][] ffnOut;
    private double[][][] output;

    public TransformerDecoderBlock(
            MultiHeadSelfAttention maskedSelfAttn,
            LayerNorm ln1,
            int embedDim,
            int numHeads,
            GradOptimizer optimizer,
            FeedForwardLayer ffn,
            LayerNorm ln2,
            LayerNorm ln3) {

        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        this.maskedSelfAttn = maskedSelfAttn;
        this.ln1 = ln1;
        this.ffn = ffn;
        this.ln2 = ln2;
        this.ln3 = ln3;

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.scale = 1.0 / Math.sqrt(headDim);
        this.optimizer = optimizer;

        // Initialize cross-attention weights
        this.Wq_cross = new double[embedDim][embedDim];
        this.Wk_cross = new double[embedDim][embedDim];
        this.Wv_cross = new double[embedDim][embedDim];
        this.Wo_cross = new double[embedDim][embedDim];

        this.gradWq_cross = new double[embedDim][embedDim];
        this.gradWk_cross = new double[embedDim][embedDim];
        this.gradWv_cross = new double[embedDim][embedDim];
        this.gradWo_cross = new double[embedDim][embedDim];

        initializeWeights();
    }

    private void initializeWeights() {
        double limit = Math.sqrt(6.0 / (embedDim + embedDim));
        for (double[][] w : new double[][][]{Wq_cross, Wk_cross, Wv_cross, Wo_cross}) {
            for (int i = 0; i < embedDim; i++) {
                for (int j = 0; j < embedDim; j++) {
                    w[i][j] = (Math.random() * 2 - 1) * limit;
                }
            }
        }
    }

    // ================== FORWARD ==================
    public double[][][] forwardBatch(double[][][] decoderInput, double[][][] encoderOutput) {
        this.decoderInput = decoderInput;
        this.encoderOutput = encoderOutput;

        int B = decoderInput.length;
        int T_dec = decoderInput[0].length;
        int T_enc = encoderOutput[0].length;
        int D = embedDim;

        // --- 1. Masked Self-Attention (Pre-LN) ---
        this.ln1Out = reshape3D(ln1.forwardBatch(flatten(decoderInput)), B, T_dec, D);
        this.selfAttnOut = maskedSelfAttn.forwardBatch(ln1Out);

        this.add1 = new double[B][T_dec][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int d = 0; d < D; d++) {
                    add1[b][t][d] = decoderInput[b][t][d] + selfAttnOut[b][t][d];
                }
            }
        }

        // --- 2. Cross-Attention (Pre-LN) ---
        this.ln2Out = reshape3D(ln2.forwardBatch(flatten(add1)), B, T_dec, D);

        // Project Q from decoder, K/V from encoder
        this.q_cross = project(ln2Out, Wq_cross);           // [B, H, T_dec, Dh]
        this.k_cross = project(encoderOutput, Wk_cross);    // [B, H, T_enc, Dh]
        this.v_cross = project(encoderOutput, Wv_cross);    // [B, H, T_enc, Dh]

        // Compute attention scores and weights
        this.attn_weights_cross = new double[B][numHeads][T_dec][T_enc];
        double[][][][] context = new double[B][numHeads][T_dec][headDim];

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < T_dec; i++) {
                    double maxScore = Double.NEGATIVE_INFINITY;
                    // Compute scores and find max for numerical stability
                    for (int j = 0; j < T_enc; j++) {
                        double score = 0;
                        for (int d = 0; d < headDim; d++) {
                            score += q_cross[b][h][i][d] * k_cross[b][h][j][d];
                        }
                        score *= scale;
                        attn_weights_cross[b][h][i][j] = score;
                        maxScore = Math.max(maxScore, score);
                    }

                    // Softmax
                    double sumExp = 0;
                    for (int j = 0; j < T_enc; j++) {
                        double expScore = Math.exp(attn_weights_cross[b][h][i][j] - maxScore);
                        attn_weights_cross[b][h][i][j] = expScore;
                        sumExp += expScore;
                    }
                    for (int j = 0; j < T_enc; j++) {
                        attn_weights_cross[b][h][i][j] /= sumExp;
                    }

                    // Weighted sum over values
                    for (int d = 0; d < headDim; d++) {
                        double val = 0;
                        for (int j = 0; j < T_enc; j++) {
                            val += attn_weights_cross[b][h][i][j] * v_cross[b][h][j][d];
                        }
                        context[b][h][i][d] = val;
                    }
                }
            }
        }

        // Concat heads and apply output projection
        this.context_concat = concatHeads(context); // [B, T_dec, D]
        this.crossAttnOut = new double[B][T_dec][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int i = 0; i < D; i++) {
                    double val = 0;
                    for (int j = 0; j < D; j++) {
                        val += context_concat[b][t][j] * Wo_cross[j][i];
                    }
                    crossAttnOut[b][t][i] = val;
                }
            }
        }

        // Residual connection
        this.add2 = new double[B][T_dec][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int d = 0; d < D; d++) {
                    add2[b][t][d] = add1[b][t][d] + crossAttnOut[b][t][d];
                }
            }
        }

        // --- 3. Feed-Forward (Pre-LN) ---
        this.ln3Out = reshape3D(ln3.forwardBatch(flatten(add2)), B, T_dec, D);
        this.ffnOut = reshape3D(ffn.forwardBatch(flatten(ln3Out)), B, T_dec, D);

        this.output = new double[B][T_dec][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int d = 0; d < D; d++) {
                    output[b][t][d] = add2[b][t][d] + ffnOut[b][t][d];
                }
            }
        }

        return output;
    }

    // ================== BACKWARD ==================
    /**
     * @param gradOutput gradient w.r.t. output [B, T_dec, D]
     * @param gradEncoderOutput (output parameter) gradient w.r.t. encoderOutput [B, T_enc, D]
     * @return gradient w.r.t. decoderInput [B, T_dec, D]
     */
    public double[][][] backwardBatch(double[][][] gradOutput, double[][][] gradEncoderOutput) {
        int B = gradOutput.length;
        int T_dec = gradOutput[0].length;
        int T_enc = encoderOutput[0].length;
        int D = embedDim;

        // --- Step 1: FFN residual backward ---
        double[][][] gradAdd2 = copy(gradOutput);
        double[][][] gradFfnOut = copy(gradOutput);

        double[][] gradFlatFfn = ffn.backwardBatch(flatten(gradFfnOut));
        double[][][] gradLn3Out = reshape3D(gradFlatFfn, B, T_dec, D);

        double[][] gradFlatAdd2FromLN3 = ln3.backwardBatch(flatten(gradLn3Out));
        double[][][] gradAdd2FromLN3 = reshape3D(gradFlatAdd2FromLN3, B, T_dec, D);

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int d = 0; d < D; d++) {
                    gradAdd2[b][t][d] += gradAdd2FromLN3[b][t][d];
                }
            }
        }

        // --- Step 2: Cross-Attention residual backward ---
        double[][][] gradAdd1 = copy(gradAdd2);
        double[][][] gradCrossAttnOut = copy(gradAdd2);

        // ---- Cross-Attention Backward ----
        // 2.1 Grad to Wo_cross
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int i = 0; i < D; i++) {
                    for (int j = 0; j < D; j++) {
                        gradWo_cross[j][i] += context_concat[b][t][j] * gradCrossAttnOut[b][t][i];
                    }
                }
            }
        }

        // 2.2 Grad to context_concat
        double[][][] gradContextConcat = new double[B][T_dec][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int j = 0; j < D; j++) {
                    double val = 0;
                    for (int i = 0; i < D; i++) {
                        val += gradCrossAttnOut[b][t][i] * Wo_cross[j][i];
                    }
                    gradContextConcat[b][t][j] = val;
                }
            }
        }

        // 2.3 Split heads
        double[][][][] gradContext = splitHeads(gradContextConcat); // [B, H, T_dec, Dh]

        // 2.4 Attention backward → gradQ, gradK, gradV
        double[][][][] gradQ = new double[B][numHeads][T_dec][headDim];
        double[][][][] gradK = new double[B][numHeads][T_enc][headDim];
        double[][][][] gradV = new double[B][numHeads][T_enc][headDim];

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < T_dec; i++) {
                    // Grad to V
                    for (int j = 0; j < T_enc; j++) {
                        for (int d = 0; d < headDim; d++) {
                            gradV[b][h][j][d] += attn_weights_cross[b][h][i][j] * gradContext[b][h][i][d];
                        }
                    }

                    // Grad to attention weights
                    double[] gradAttn = new double[T_enc];
                    for (int j = 0; j < T_enc; j++) {
                        for (int d = 0; d < headDim; d++) {
                            gradAttn[j] += gradContext[b][h][i][d] * v_cross[b][h][j][d];
                        }
                    }

                    double sumGradAttn = 0;
                    for (int j = 0; j < T_enc; j++) {
                        sumGradAttn += gradAttn[j] * attn_weights_cross[b][h][i][j];
                    }

                    for (int j = 0; j < T_enc; j++) {
                        double gradScore = attn_weights_cross[b][h][i][j] * (gradAttn[j] - sumGradAttn);
                        gradScore *= scale;

                        for (int d = 0; d < headDim; d++) {
                            gradQ[b][h][i][d] += gradScore * k_cross[b][h][j][d];
                            gradK[b][h][j][d] += gradScore * q_cross[b][h][i][d];
                        }
                    }
                }
            }
        }

        // 2.5 Project gradients back to input space
        double[][][] gradLn2Out = new double[B][T_dec][D];
        double[][][] gradEncoderTemp = new double[B][T_enc][D]; // accumulate K and V grads

        projectBackward(gradLn2Out, gradQ, Wq_cross);
        projectBackward(gradEncoderTemp, gradK, Wk_cross);
        projectBackward(gradEncoderTemp, gradV, Wv_cross);

        // Accumulate weight gradients
        accumulateGrad(Wq_cross, ln2Out, gradQ, gradWq_cross);
        accumulateGrad(Wk_cross, encoderOutput, gradK, gradWk_cross);
        accumulateGrad(Wv_cross, encoderOutput, gradV, gradWv_cross);

        // Add gradLn2Out to gradAdd1 path
        double[][] gradFlatAdd1FromLN2 = ln2.backwardBatch(flatten(gradLn2Out));
        double[][][] gradAdd1FromLN2 = reshape3D(gradFlatAdd1FromLN2, B, T_dec, D);
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int d = 0; d < D; d++) {
                    gradAdd1[b][t][d] += gradAdd1FromLN2[b][t][d];
                }
            }
        }

        // Copy encoder gradient to output
        if (gradEncoderOutput != null) {
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T_enc; t++) {
                    System.arraycopy(gradEncoderTemp[b][t], 0, gradEncoderOutput[b][t], 0, D);
                }
            }
        }

        // --- Step 3: Self-Attention residual backward ---
        double[][][] gradDecoderInput = copy(gradAdd1);
        double[][][] gradSelfAttnOut = copy(gradAdd1);

        double[][][] gradLn1Out = maskedSelfAttn.backwardBatch(gradSelfAttnOut);
        double[][] gradFlatDecoderInputFromLN1 = ln1.backwardBatch(flatten(gradLn1Out));
        double[][][] gradDecoderInputFromLN1 = reshape3D(gradFlatDecoderInputFromLN1, B, T_dec, D);

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T_dec; t++) {
                for (int d = 0; d < D; d++) {
                    gradDecoderInput[b][t][d] += gradDecoderInputFromLN1[b][t][d];
                }
            }
        }

        return gradDecoderInput;
    }

    // ================== PARAMETER UPDATE ==================
    @Override
    public void updateParameters() {
        maskedSelfAttn.updateParameters();
        ffn.updateParameters();
        ln1.updateParameters();
        ln2.updateParameters();
        ln3.updateParameters();

        optimizer.applyGradients(Wq_cross, gradWq_cross);
        optimizer.applyGradients(Wk_cross, gradWk_cross);
        optimizer.applyGradients(Wv_cross, gradWv_cross);
        optimizer.applyGradients(Wo_cross, gradWo_cross);

        clearGradients();
    }

    private void clearGradients() {
        for (double[][] g : new double[][][]{gradWq_cross, gradWk_cross, gradWv_cross, gradWo_cross}) {
            for (double[] row : g) Arrays.fill(row, 0.0);
        }
    }

    // ================== UTILITIES ==================
    private double[][][][] project(double[][][] input, double[][] weight) {
        int B = input.length;
        int T = input[0].length;
        double[][][][] out = new double[B][numHeads][T][headDim];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < embedDim; d++) {
                    double val = 0;
                    for (int k = 0; k < embedDim; k++) {
                        val += input[b][t][k] * weight[k][d];
                    }
                    int h = d / headDim;
                    int hd = d % headDim;
                    out[b][h][t][hd] = val;
                }
            }
        }
        return out;
    }

    private double[][][] concatHeads(double[][][][] context) {
        int B = context.length;
        int T = context[0][0].length;
        double[][][] out = new double[B][T][embedDim];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        out[b][t][h * headDim + d] = context[b][h][t][d];
                    }
                }
            }
        }
        return out;
    }

    private double[][][][] splitHeads(double[][][] input) {
        int B = input.length;
        int T = input[0].length;
        double[][][][] out = new double[B][numHeads][T][headDim];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < embedDim; d++) {
                    int h = d / headDim;
                    int hd = d % headDim;
                    out[b][h][t][hd] = input[b][t][d];
                }
            }
        }
        return out;
    }

    private void projectBackward(double[][][] gradInput, double[][][][] gradProj, double[][] weight) {
        int B = gradInput.length;
        int T = gradInput[0].length;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < embedDim; d++) {
                    int h = d / headDim;
                    int hd = d % headDim;
                    double g = gradProj[b][h][t][hd];
                    for (int k = 0; k < embedDim; k++) {
                        gradInput[b][t][k] += g * weight[k][d];
                    }
                }
            }
        }
    }

    private void accumulateGrad(double[][] weight, double[][][] input, double[][][][] gradProj, double[][] gradWeight) {
        int B = input.length;
        int T = input[0].length;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < embedDim; d++) {
                    int h = d / headDim;
                    int hd = d % headDim;
                    double go = gradProj[b][h][t][hd];
                    for (int k = 0; k < embedDim; k++) {
                        gradWeight[k][d] += input[b][t][k] * go;
                    }
                }
            }
        }
    }

    private static double[][] flatten(double[][][] x) {
        int B = x.length, T = x[0].length, D = x[0][0].length;
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
        int B = src.length, T = src[0].length, D = src[0][0].length;
        double[][][] dst = new double[B][T][D];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                System.arraycopy(src[b][t], 0, dst[b][t], 0, D);
            }
        }
        return dst;
    }
}