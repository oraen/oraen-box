package com.oraen.box.otorch.optimizer;

import com.oraen.box.otorch.GradOptimizer;
import lombok.Data;
import lombok.Setter;

/**
 * AdamW优化器 - Adam with decoupled weight decay
 *
 * 特点：
 * 1. 继承Adam的自适应学习率特性
 * 2. 将权重衰减（L2正则化）与梯度更新解耦
 * 3. 数学形式：θ_t = θ_{t-1} - η·[m̂_t/(√v̂_t+ε) + λ·θ_{t-1}]
 *
 * 参考文献：
 * - Adam: Kingma & Ba, "Adam: A Method for Stochastic Optimization", 2014
 * - AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", 2017
 */
@Data
public class AdamWOptimizer implements GradOptimizer {

    // ====================== 超参数 ======================

    /** 基础学习率
     * -- SETTER --
     *  设置学习率（支持学习率调度）
     */
    @Setter
    private double learningRate;

    /** 输出维度（神经元数量） */
    private final int outputDim;

    /** 输入维度（特征数量） */
    private final int inputDim;

    /**
     * 一阶动量衰减系数 β₁
     * 控制梯度方向历史的平滑程度
     * 默认值：0.9
     */
    private double beta1;

    /**
     * 二阶动量衰减系数 β₂
     * 控制梯度平方历史的平滑程度
     * 默认值：0.999
     */
    private double beta2;

    /**
     * 解耦权重衰减系数 λ (lambda)
     * 注意：这里通常使用较小的值，如 0.01、0.001
     * 作用：直接作用于参数本身，与学习率解耦
     * -- SETTER --
     *  设置权重衰减系数

     */
    @Setter
    private double weightDecay;

    /**
     * 数值稳定常数 ε
     * 防止除以零，保持数值稳定性
     * 默认值：1e-8
     */
    private double eps;

    // ====================== 状态变量 ======================

    /** 权重一阶动量（方向历史） */
    private double[][] mW;

    /** 权重二阶动量（梯度平方历史） */
    private double[][] vW;

    /** 偏置一阶动量 */
    private double[] mB;

    /** 偏置二阶动量 */
    private double[] vB;

    /** 权重更新步数（独立计数器） */
    private int tW = 0;

    /** 偏置更新步数（独立计数器） */
    private int tB = 0;

    /** β₁^tW，用于权重偏差修正 */
    private double beta1PowTw = 1.0;

    /** β₂^tW，用于权重偏差修正 */
    private double beta2PowTw = 1.0;

    /** β₁^tB，用于偏置偏差修正 */
    private double beta1PowTb = 1.0;

    /** β₂^tB，用于偏置偏差修正 */
    private double beta2PowTb = 1.0;

    // ====================== 构造函数 ======================

    /**
     * 完整参数构造函数
     *
     * @param outputDim 输出维度
     * @param inputDim 输入维度
     * @param lr 学习率
     * @param b1 β₁，一阶动量衰减系数
     * @param b2 β₂，二阶动量衰减系数
     * @param wd 权重衰减系数 λ
     * @param eps 数值稳定常数
     */
    public AdamWOptimizer(int outputDim, int inputDim, double lr, double b1, double b2, double wd, double eps) {
        this.outputDim = outputDim;
        this.inputDim = inputDim;
        this.learningRate = lr;
        this.beta1 = b1;
        this.beta2 = b2;
        this.weightDecay = wd;
        this.eps = eps;

        // 初始化动量状态为零
        mW = new double[outputDim][inputDim];
        vW = new double[outputDim][inputDim];
        mB = new double[outputDim];
        vB = new double[outputDim];
    }

    /**
     * 常用参数构造函数
     * 默认：lr=0.001, β₁=0.9, β₂=0.999, λ=0.01, ε=1e-8
     */
    public AdamWOptimizer(int outputDim, int inputDim) {
        this(outputDim, inputDim, 0.001, 0.9, 0.999, 0.01, 1e-8);
    }

    /**
     * 带学习率和权重衰减的构造函数
     */
    public AdamWOptimizer(int outputDim, int inputDim, double lr, double wd) {
        this(outputDim, inputDim, lr, 0.9, 0.999, wd, 1e-8);
    }

    // ====================== 核心方法 ======================

    @Override
    public void applyGradients(double[][] weight, double[][] gradWeight) {
        tW++;

        // 更新幂值：β₁^tW 和 β₂^tW（通过累积乘法）
        beta1PowTw *= beta1;
        beta2PowTw *= beta2;

        // 偏差修正分母：1 - β^t，防止除零
        double biasCorrection1 = Math.max(1 - beta1PowTw, eps);
        double biasCorrection2 = Math.max(1 - beta2PowTw, eps);

        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {

                // ========== Adam部分：自适应学习率更新 ==========

                // 1. 更新一阶动量：m_t = β₁·m_{t-1} + (1-β₁)·g_t
                mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * gradWeight[i][j];

                // 2. 更新二阶动量：v_t = β₂·v_{t-1} + (1-β₂)·g_t²
                vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * gradWeight[i][j] * gradWeight[i][j];

                // 3. 偏差修正：m̂ = m / (1-β₁ᵗ), v̂ = v / (1-β₂ᵗ)
                double mHat = mW[i][j] / biasCorrection1;
                double vHat = vW[i][j] / biasCorrection2;

                // 4. Adam更新量：η·m̂/(√v̂+ε)
                double adamUpdate = learningRate * mHat / (Math.sqrt(vHat) + eps);

                // ========== AdamW部分：解耦权重衰减 ==========

                // 5. 权重衰减项：η·λ·θ
                double decayUpdate = learningRate * weightDecay * weight[i][j];

                // 6. 组合更新：θ = θ - [Adam更新 + 权重衰减]
                weight[i][j] -= adamUpdate + decayUpdate;

                // 数学形式：θ_t = θ_{t-1} - η·[m̂_t/(√v̂_t+ε) + λ·θ_{t-1}]
            }
        }
    }

    @Override
    public void applyGradients(double[] bias, double[] gradBias) {
        tB++;

        // 更新幂值：β₁^tB 和 β₂^tB
        beta1PowTb *= beta1;
        beta2PowTb *= beta2;

        // 偏差修正分母
        double biasCorrection1 = Math.max(1 - beta1PowTb, eps);
        double biasCorrection2 = Math.max(1 - beta2PowTb, eps);

        for (int i = 0; i < outputDim; i++) {

            // ========== Adam部分 ==========

            // 1. 更新偏置一阶动量
            mB[i] = beta1 * mB[i] + (1 - beta1) * gradBias[i];

            // 2. 更新偏置二阶动量
            vB[i] = beta2 * vB[i] + (1 - beta2) * gradBias[i] * gradBias[i];

            // 3. 偏差修正
            double mHat = mB[i] / biasCorrection1;
            double vHat = vB[i] / biasCorrection2;

            // 4. Adam更新量
            double adamUpdate = learningRate * mHat / (Math.sqrt(vHat) + eps);

            // ========== AdamW部分 ==========

            // 5. 权重衰减项（偏置通常不加权重衰减，但这里为了统一性保留）
            // 注意：实际应用中偏置是否加权重衰减是可选的
            double decayUpdate = learningRate * weightDecay * bias[i];

            // 6. 组合更新
            bias[i] -= adamUpdate + decayUpdate;
        }
    }


}