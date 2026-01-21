package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Learnable;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

/**
 * 多头自注意力（Multi-Head Self Attention）with RoPE
 *
 * 输入:  [B, T, D]
 * 输出:  [B, T, D]
 *
 * 其中:
 * B = batch size
 * T = 序列长度
 * D = embedDim
 * H = numHeads
 * Dh = headDim = D / H
 */
@Getter
public class MultiHeadSelfAttention implements AttentionLayer, Learnable {

    private final int embedDim;     // 总 embedding 维度 D
    private final int numHeads;     // 头数 H
    private final int headDim;      // 每个 head 的维度 Dh = D / H
    private final double scale;     // 1 / sqrt(Dh)

    // ================= 可学习参数 =================

    // Wq, Wk, Wv: [D, D] - 用于生成 Q, K, V
    // Wo: [D, D] - 用于合并多头输出
    private final double[][] Wq;
    private final double[][] Wk;
    private final double[][] Wv;
    private final double[][] Wo;

    // 梯度
    private final double[][] gradWq;
    private final double[][] gradWk;
    private final double[][] gradWv;
    private final double[][] gradWo;

    @Setter
    private GradOptimizer optimizer;

    // ================= forward 缓存（用于 backward）=================

    private double[][][] input;        // [B, T, D]
    private double[][][][] q;          // [B, H, T, Dh]
    private double[][][][] k;          // [B, H, T, Dh]
    private double[][][][] v;          // [B, H, T, Dh]
    private double[][][][] attn;       // [B, H, T, T] - attention 权重
    private double[][][] concatContext; // [B, T, D] - 合并后的 context（Wo 之前）

    public MultiHeadSelfAttention(int embedDim, int numHeads, GradOptimizer optimizer) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim 必须能被 numHeads 整除");
        }

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        //headDim 一定是偶数
        this.headDim = embedDim / numHeads;
        this.scale = 1.0 / Math.sqrt(headDim);
        this.optimizer = optimizer;

        this.Wq = new double[embedDim][embedDim];
        this.Wk = new double[embedDim][embedDim];
        this.Wv = new double[embedDim][embedDim];
        this.Wo = new double[embedDim][embedDim];

        this.gradWq = new double[embedDim][embedDim];
        this.gradWk = new double[embedDim][embedDim];
        this.gradWv = new double[embedDim][embedDim];
        this.gradWo = new double[embedDim][embedDim];

        // TODO: 使用适当的初始化器（如 Xavier/He 初始化）
        initializeWeights();
    }

    private void initializeWeights() {
        // 简单的 Xavier 初始化
        double limit = Math.sqrt(6.0 / (embedDim + embedDim));
        for (double[][] w : new double[][][]{Wq, Wk, Wv, Wo}) {
            for (int i = 0; i < embedDim; i++) {
                for (int j = 0; j < embedDim; j++) {
                    w[i][j] = (Math.random() * 2 - 1) * limit;
                }
            }
        }
    }

    // =========================================================
    // Forward Pass
    // =========================================================
    public double[][][] forwardBatch(double[][][] data) {
        this.input = data;

        int B = data.length;
        int T = data[0].length;

        // 1. 线性投影生成 Q, K, V: [B, T, D] -> [B, H, T, Dh]
        q = project(data, Wq);
        k = project(data, Wk);
        v = project(data, Wv);

        // 2. 应用 RoPE（旋转位置编码）到 Q 和 K
        applyRoPE(q);
        applyRoPE(k);

        // 3. 计算注意力分数并应用 softmax
        attn = new double[B][numHeads][T][T];
        double[][][][] context = new double[B][numHeads][T][headDim];

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < T; i++) {

                    // 计算 attention score: Q * K^T
                    double max = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < T; j++) {
                        double score = 0;
                        for (int d = 0; d < headDim; d++) {
                            score += q[b][h][i][d] * k[b][h][j][d];
                        }
                        score *= scale;
                        attn[b][h][i][j] = score;
                        max = Math.max(max, score);
                    }

                    // Softmax（数值稳定版本）
                    double sum = 0;
                    for (int j = 0; j < T; j++) {
                        attn[b][h][i][j] = Math.exp(attn[b][h][i][j] - max);
                        sum += attn[b][h][i][j];
                    }
                    for (int j = 0; j < T; j++) {
                        attn[b][h][i][j] /= sum;
                    }

                    // 计算加权和: context = attn * V
                    for (int d = 0; d < headDim; d++) {
                        double val = 0;
                        for (int j = 0; j < T; j++) {
                            val += attn[b][h][i][j] * v[b][h][j][d];
                        }
                        context[b][h][i][d] = val;
                    }
                }
            }
        }

        // 4. 合并多头并通过 Wo 投影
        return combine(context);
    }

    // =========================================================
    // Backward Pass
    // =========================================================
    public double[][][] backwardBatch(double[][][] gradOutput) {
        int B = gradOutput.length;
        int T = gradOutput[0].length;

        double[][][] gradInput = new double[B][T][embedDim];

        // -------- Step 1: Wo 的反向传播 --------

        // gradWo 累加
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int i = 0; i < embedDim; i++) {
                    for (int j = 0; j < embedDim; j++) {
                        gradWo[j][i] += concatContext[b][t][j] * gradOutput[b][t][i];
                    }
                }
            }
        }

        // gradConcat = gradOutput * Wo^T
        double[][][] gradConcat = new double[B][T][embedDim];
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int j = 0; j < embedDim; j++) {
                    double val = 0;
                    for (int i = 0; i < embedDim; i++) {
                        val += gradOutput[b][t][i] * Wo[j][i];
                    }
                    gradConcat[b][t][j] = val;
                }
            }
        }

        // -------- Step 2: 将 gradConcat 拆分成多头 --------
        double[][][][] gradContext = splitHeads(gradConcat);

        // -------- Step 3: Attention 的反向传播 --------
        double[][][][] gradQ = new double[B][numHeads][T][headDim];
        double[][][][] gradK = new double[B][numHeads][T][headDim];
        double[][][][] gradV = new double[B][numHeads][T][headDim];

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < T; i++) {

                    // gradV: 直接从 context 梯度计算
                    for (int j = 0; j < T; j++) {
                        for (int d = 0; d < headDim; d++) {
                            gradV[b][h][j][d] += attn[b][h][i][j] * gradContext[b][h][i][d];
                        }
                    }

                    // gradAttn: gradContext · V^T
                    double[] gradAttnRow = new double[T];
                    for (int j = 0; j < T; j++) {
                        for (int d = 0; d < headDim; d++) {
                            gradAttnRow[j] += gradContext[b][h][i][d] * v[b][h][j][d];
                        }
                    }

                    // Softmax 反向传播
                    double sumGradAttn = 0;
                    for (int j = 0; j < T; j++) {
                        sumGradAttn += gradAttnRow[j] * attn[b][h][i][j];
                    }

                    // gradScore = softmax 梯度
                    for (int j = 0; j < T; j++) {
                        double gradScore = attn[b][h][i][j] * (gradAttnRow[j] - sumGradAttn);
                        gradScore *= scale; // 别忘了 scale

                        // gradQ 和 gradK
                        for (int d = 0; d < headDim; d++) {
                            gradQ[b][h][i][d] += gradScore * k[b][h][j][d];
                            gradK[b][h][j][d] += gradScore * q[b][h][i][d];
                        }
                    }
                }
            }
        }

        // -------- Step 4: RoPE 反向传播（逆旋转）--------
        applyRoPEBackward(gradQ);
        applyRoPEBackward(gradK);

        // -------- Step 5: 投影层的反向传播 --------
        // 计算 Wq, Wk, Wv 的梯度
        accumulateGradProj(gradWq, input, gradQ);
        accumulateGradProj(gradWk, input, gradK);
        accumulateGradProj(gradWv, input, gradV);

        // 计算对输入的梯度
        projectBackward(gradInput, gradQ, Wq);
        projectBackward(gradInput, gradK, Wk);
        projectBackward(gradInput, gradV, Wv);

        return gradInput;
    }

    public void updateParameters() {
        optimizer.applyGradients(Wq, gradWq);
        optimizer.applyGradients(Wk, gradWk);
        optimizer.applyGradients(Wv, gradWv);
        optimizer.applyGradients(Wo, gradWo);
        clearGrad();
    }

    private void clearGrad() {
        for (double[][] g : new double[][][]{gradWq, gradWk, gradWv, gradWo}) {
            for (double[] row : g) {
                Arrays.fill(row, 0);
            }
        }
    }

    // =========================================================
    // RoPE (Rotary Position Embedding)
    // =========================================================
    private void applyRoPE(double[][][][] x) {
        rotate(x, false);
    }

    private void applyRoPEBackward(double[][][][] grad) {
        rotate(grad, true);
    }

    /**
     * RoPE 旋转操作
     *
     * 标准 RoPE 公式：对于第 pair 对维度（索引 2*pair 和 2*pair+1）
     * theta_pair = 10000^(-2*pair/d) = 1 / 10000^(2*pair/d)
     *
     * @param x 输入张量 [B, H, T, Dh]
     * @param inverse 是否为逆旋转（用于反向传播）
     */
    private void rotate(double[][][][] x, boolean inverse) {
        int B = x.length;
        int T = x[0][0].length;

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int t = 0; t < T; t++) {
                    // 每次处理一对维度
                    for (int pair = 0; pair < headDim / 2; pair++) {
                        int i = pair * 2;  // 实际维度索引

                        // 正确的 RoPE 公式：theta_pair = 10000^(-2*pair/d)
                        double theta = 1.0 / Math.pow(10000.0, (2.0 * pair) / headDim);
                        double angle = t * theta;

                        double cos = Math.cos(angle);
                        double sin = inverse ? -Math.sin(angle) : Math.sin(angle);

                        double x1 = x[b][h][t][i];
                        double x2 = x[b][h][t][i + 1];

                        // 旋转矩阵：[cos -sin] [x1]
                        //          [sin  cos] [x2]
                        x[b][h][t][i]     = x1 * cos - x2 * sin;
                        x[b][h][t][i + 1] = x1 * sin + x2 * cos;
                    }
                }
            }
        }
    }

    // =========================================================
    // 投影和合并操作
    // =========================================================

    /**
     * 线性投影并拆分成多头
     *
     * @param input  [B, T, D]
     * @param weight [D, D]
     * @return [B, H, T, Dh]
     */
    private double[][][][] project(double[][][] input, double[][] weight) {
        int B = input.length;
        int T = input[0].length;

        double[][][][] out = new double[B][numHeads][T][headDim];

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // 矩阵乘法: input[b][t] @ weight
                for (int d = 0; d < embedDim; d++) {
                    double val = 0;
                    for (int k = 0; k < embedDim; k++) {
                        val += input[b][t][k] * weight[k][d];
                    }

                    // 拆分到对应的 head
                    int h = d / headDim;
                    int hd = d % headDim;
                    out[b][h][t][hd] = val;
                }
            }
        }
        return out;
    }

    /**
     * 合并多头并通过 Wo 投影
     *
     * @param context [B, H, T, Dh]
     * @return [B, T, D]
     */
    private double[][][] combine(double[][][][] context) {
        int B = context.length;
        int T = context[0][0].length;

        // 缓存 concat 结果用于反向传播
        concatContext = new double[B][T][embedDim];
        double[][][] output = new double[B][T][embedDim];

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {

                // 合并所有 head: [H, Dh] -> [D]
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        concatContext[b][t][h * headDim + d] = context[b][h][t][d];
                    }
                }

                // Wo 投影: concat @ Wo
                for (int i = 0; i < embedDim; i++) {
                    double val = 0;
                    for (int j = 0; j < embedDim; j++) {
                        val += concatContext[b][t][j] * Wo[j][i];
                    }
                    output[b][t][i] = val;
                }
            }
        }
        return output;
    }

    /**
     * 将 [B, T, D] 拆分成多头 [B, H, T, Dh]
     */
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

    // =========================================================
    // 梯度计算辅助函数
    // =========================================================

    /**
     * 累加投影层参数梯度
     *
     * @param gradWeight [D, D] - 参数梯度累加器
     * @param input      [B, T, D] - 前向传播的输入
     * @param gradOut    [B, H, T, Dh] - 来自上层的梯度
     */
    private void accumulateGradProj(
            double[][] gradWeight,
            double[][][] input,
            double[][][][] gradOut
    ) {
        int B = input.length;
        int T = input[0].length;

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < embedDim; d++) {
                    int h = d / headDim;
                    int hd = d % headDim;
                    double go = gradOut[b][h][t][hd];

                    for (int k = 0; k < embedDim; k++) {
                        gradWeight[k][d] += input[b][t][k] * go;
                    }
                }
            }
        }
    }

    /**
     * 将梯度从投影后的张量回传到输入
     *
     * @param gradInput [B, T, D] - 累加输入梯度
     * @param gradProj  [B, H, T, Dh] - 投影后的梯度
     * @param weight    [D, D] - 投影权重
     */
    private void projectBackward(
            double[][][] gradInput,
            double[][][][] gradProj,
            double[][] weight
    ) {
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
}