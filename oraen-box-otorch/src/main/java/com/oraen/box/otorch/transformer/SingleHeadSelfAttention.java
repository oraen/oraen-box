package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.activation.ActivationLayerAdapter;
import com.oraen.box.otorch.activation.ScaleActivationFunction;
import com.oraen.box.otorch.component.SoftmaxLayer;
import com.oraen.box.otorch.util.DataUtil;
import com.oraen.box.otorch.util.TensorUtil;
import lombok.Getter;
import lombok.Setter;


@Getter
public class SingleHeadSelfAttention implements AttentionLayer, Learnable {

    private final int inputDim;     // 输入维度
    private final int outputDim;     // 输出维度，如果要组成多注意力头，一般取inputDim/头数
    private final double scale;     // 缓存 1 / sqrt(outputDim)，提高计算效率

    // ================= 可学习参数 =================

    // Wq, Wk, Wv: [Dout, Din] - 用于生成 Q, K, V
    private final double[][] Wq;
    private final double[][] Wk;
    private final double[][] Wv;

    // 梯度
    private final double[][] gradWq;
    private final double[][] gradWk;
    private final double[][] gradWv;

    @Setter
    private GradOptimizer qOptimizer;
    @Setter
    private GradOptimizer kOptimizer;
    @Setter
    private GradOptimizer vOptimizer;

    private Layer<double[], double[]> scaleLayer;
    private final SoftmaxLayer softmaxLayer = new SoftmaxLayer();


    // ================= forward 缓存（用于 backward）=================
    private double[][][] input;        // [B, T, D]
    private double[][][] attn;       // [B, T, T] - attention 权重

    private double[][][] qCache;
    private double[][][] kCache;
    private double[][][] vCache;

    private final PositionalEncodingType positionalEncodingType;
    private final boolean useMask;


    public SingleHeadSelfAttention(int inputDim, int outputDim, GradOptimizer qOptimizer, GradOptimizer kOptimizer, GradOptimizer vOptimizer,
                                   ParamInitializer paramInitializer, PositionalEncodingType positionalEncodingType, boolean useMask) {
        if(positionalEncodingType == PositionalEncodingType.ROPE && outputDim % 2 != 0) {
            throw new IllegalArgumentException("outputDim must be even when using ROPE");
        }

        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.scale = 1.0 / Math.sqrt(outputDim);
        this.scaleLayer = new ActivationLayerAdapter(new ScaleActivationFunction(scale));

        this.qOptimizer = qOptimizer;
        this.kOptimizer = kOptimizer;
        this.vOptimizer = vOptimizer;

        this.Wq = new double[outputDim][inputDim];
        this.Wk = new double[outputDim][inputDim];
        this.Wv = new double[outputDim][inputDim];

        this.gradWq = new double[outputDim][inputDim];
        this.gradWk = new double[outputDim][inputDim];
        this.gradWv = new double[outputDim][inputDim];

        this.positionalEncodingType = positionalEncodingType;
        this.useMask = useMask;


        // 权重初始化
        paramInitializer.initializeWeights(Wq);
        paramInitializer.initializeWeights(Wk);
        paramInitializer.initializeWeights(Wv);
    }


    @Override
    public double[][][] forwardBatch(double[][][] data) {
        this.input = data;

        int batchSize = data.length;
        attn = new double[batchSize][][];
        double[][][] out = new double[batchSize][][];
        qCache = new double[batchSize][][];
        kCache = new double[batchSize][][];
        vCache = new double[batchSize][][];
        for (int b = 0; b < batchSize; b++) {
            out[b] = forward0(data[b], b);
        }
        return out;
    }


    public double[][][] backwardBatch(double[][][] gradOutput) {
        int batchSize = gradOutput.length;
        double[][][] gradInput = new double[batchSize][][];

        for (int b = 0; b < batchSize; b++) {
            gradInput[b] = backward0(gradOutput[b], b);
        }

        return gradInput;
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

    public void updateParameters() {
        qOptimizer.applyGradients(Wq, gradWq);
        kOptimizer.applyGradients(Wk, gradWk);
        vOptimizer.applyGradients(Wv, gradWv);
        clearGrad();
    }

    private void clearGrad() {
        DataUtil.clear(gradWq, gradWk, gradWv);
    }

    private void applyRoPE(double[][] x) {
        rotate(x, false);
    }

    private void applyRoPEBackward(double[][] grad) {
        rotate(grad, true);
    }



    public double[][] forward0(double[][] data, int batchIndex) {
        double[][] q = TensorUtil.project(data, Wq);
        double[][] k = TensorUtil.project(data, Wk);
        double[][] v = TensorUtil.project(data, Wv);

        if(positionalEncodingType == PositionalEncodingType.ROPE) {
            applyRoPE(q);
            applyRoPE(k);
        }

        qCache[batchIndex] = q;
        kCache[batchIndex] = k;
        vCache[batchIndex] = v;

        double[][] theAttn = TensorUtil.project(q, k);
        if (useMask) {
            applyCausalMask(theAttn);
        }

        theAttn = scaleLayer.forwardBatch(theAttn);
        theAttn = softmaxLayer.forwardBatch(theAttn);
        attn[batchIndex] = theAttn;
        return TensorUtil.matmul(theAttn, v);
    }

    public double[][] backward0(double[][] gradOutput, int batchIndex) {
        int seqLen = gradOutput.length;
        double[][] q = this.qCache[batchIndex];             // [T, D_out] (rotated if ROPE)
        double[][] k = this.kCache[batchIndex];             // [T, D_out] (rotated if ROPE)
        double[][] v = this.vCache[batchIndex];             // [T, D_out]
        double[][] input = this.input[batchIndex];              // [T, inputDim]
        double[][] attn = this.attn[batchIndex];            // [T, T]
        double[][] gradInput = new double[seqLen][inputDim];


        // Step 1: gradV = attn^T @ gOut
        double[][] gradV = TensorUtil.matmul(TensorUtil.transpose(attn), gradOutput); // [T, D_out]

        // Step 2: gradAttn = gOut @ v^T
        double[][] gradAttn = TensorUtil.project(gradOutput, v); // [T, T]

        // Step 3: Softmax 反向
        double[][] gradSoftmaxInput = softmaxLayer.backwardBatch(gradAttn); // [T, T]

        // Step 4: Scale 反向 (乘以 scale)
        double[][] gradScores = scaleLayer.backwardBatch(gradSoftmaxInput);

        // Step 5: gradQ = gradScores @ k
        double[][] gradQ = TensorUtil.matmul(gradScores, k); // [T, D_out]

        // Step 6: gradK = gradScores^T @ q
        double[][] gradK = TensorUtil.matmul(TensorUtil.transpose(gradScores), q); // [T, D_out]

        // Step 7: 如果使用 ROPE，需要反向旋转 gradQ 和 gradK
        if (positionalEncodingType == PositionalEncodingType.ROPE) {
            applyRoPEBackward(gradQ);
            applyRoPEBackward(gradK);
        }

        // Step 8: 计算权重梯度 (gradW += gradAct^T @ input)
        accumulateGrad(gradWq, gradQ, input);
        accumulateGrad(gradWk, gradK, input);
        accumulateGrad(gradWv, gradV, input);

        // Step 9: 计算输入梯度: gradInput = gradQ @ Wq + gradK @ Wk + gradV @ Wv
        double[][] gInputQ = TensorUtil.matmul(gradQ, Wq); // [T, inputDim]
        double[][] gInputK = TensorUtil.matmul(gradK, Wk); // [T, inputDim]
        double[][] gInputV = TensorUtil.matmul(gradV, Wv); // [T, inputDim]

        for (int t = 0; t < seqLen; t++) {
            for (int d = 0; d < inputDim; d++) {
                gradInput[t][d] = gInputQ[t][d] + gInputK[t][d] + gInputV[t][d];
            }
        }


        return gradInput;
    }




    /**
     * RoPE 原地旋转操作，使用RoPE做位置编码
     *
     * 标准 RoPE 公式：对于第 pair 对维度（索引 2*pair 和 2*pair+1）
     * theta_pair = 10000^(-2*pair/d) = 1 / 10000^(2*pair/d)
     *
     * @param inverse 是否为逆旋转（用于反向传播），正常为false时是逆时针旋转，true时为顺时针旋转
     */
    private void rotate(double[][] x, boolean inverse) {
        int seqLen = x.length;
        for (int t = 0; t < seqLen; t++) {
            // 每次处理一对维度
            for (int pair = 0; pair < outputDim / 2; pair++) {
                int i = pair * 2;  // 实际维度索引

                // 旋转的角频率 公式：theta_pair = 10000^(-2*pair/d)，可以看到越在后面的维度频率越低
                double theta = Math.pow(10000.0, (-2.0 * pair) / outputDim);
                //每个维度旋转的角度，根据序列位置增加
                double angle = t * theta;

                //根据这个角度旋转这个维度对组成的二维向量，可以画图感受一下旋转后两个维度变量的变化
                double cos = Math.cos(angle);
                double sin = inverse ? -Math.sin(angle) : Math.sin(angle);

                double x1 = x[t][i];
                double x2 = x[t][i + 1];

                // 旋转矩阵：[cos -sin] [x1]
                //          [sin  cos] [x2]
                x[t][i]     = x1 * cos - x2 * sin;
                x[t][i + 1] = x1 * sin + x2 * cos;
            }
        }
    }

    /**
     * 应用因果掩码（Causal Mask / Look-ahead Mask）
     * 将注意力矩阵的上三角部分（不包括对角线）设为负无穷大
     * 这样在 softmax 后，这些位置的权重会变成 0
     */
    private void applyCausalMask(double[][] attn) {
        int seqLen = attn.length;
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                attn[i][j] = Double.NEGATIVE_INFINITY;
            }
        }
    }


    public enum PositionalEncodingType {
        NONE,
        ROPE
    }
}
