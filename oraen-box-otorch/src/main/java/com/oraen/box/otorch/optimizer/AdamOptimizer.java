package com.oraen.box.otorch.optimizer;

import com.oraen.box.otorch.GradOptimizer;
import lombok.Data;
import lombok.Getter;

@Data
public class AdamOptimizer implements GradOptimizer {

    //学习率
    private double learningRate;
    private final int outputDim;
    private final int inputDim;

    /**
     * beta1 / β1
     * 中文名：一阶动量衰减系数
     * 作用：控制历史梯度方向向量 m 的平滑程度
     *      越接近1 → 方向记忆越久、更新更平滑但响应变慢
     * 数学对应：m_t = β1 * m_{t-1} + (1-β1) * g_t
     */
    private double beta1;

    /**
     * beta2 / β2
     * 中文名：二阶矩衰减系数
     * 作用：对历史梯度平方 v 做指数衰减统计，用于估计梯度幅值强度
     *      决定该参数维度上的自适应学习率
     * 数学对应：v_t = β2 * v_{t-1} + (1-β2) * g_t^2
     */
    private double beta2;


    // ---------------- Adam 状态变量 ----------------

    /**
     * mW
     * 权重一阶动量（方向历史向量）
     * 作用：对每个权重位置累计历史梯度方向，减少震荡、提升稳定性
     *      不同位置独立维护，不可跨层混用, 参考 Momentum 优化器
     */
    private double[][] mW;

    /**
     * vW
     * 权重二阶矩（梯度幅值历史）
     * 作用：记录每个权重位置 g^2 的指数均值
     *      用于自适应缩小大梯度步长、放大小梯度步长， 参考 RMSProp 优化器
     */
    private double[][] vW;

    /**
     * mB
     * 偏置一阶动量
     * 作用：对应 bias 的方向历史，与 mW 同理
     */
    private double[] mB;

    /**
     * vB
     * 偏置二阶矩
     * 作用：对应 bias 的梯度平方历史，用于缩放偏置更新
     */
    private double[] vB;

    /**
     * t
     * 时间步 / step
     * 作用：用于偏差修正
     *      mHat = m/(1-β1^t)
     *      vHat = v/(1-β2^t)
     *      解决早期历史从0起步的有偏估计问题
     */
    private int t = 0;

    private double eps = 1e-8;


    public AdamOptimizer(int outputDim, int inputDim, double lr, double b1, double b2) {
        this.learningRate = lr;
        this.beta1 = b1;
        this.beta2 = b2;
        this.outputDim = outputDim;
        this.inputDim = inputDim;
        mW = new double[outputDim][inputDim];
        vW = new double[outputDim][inputDim];
        mB = new double[outputDim];
        vB = new double[outputDim];
    }

    public AdamOptimizer(int outputDim, int inputDim) {
        this(outputDim, inputDim, 0.001, 0.9, 0.999);
    }



    @Override
    public void applyGradients(double[][] weight, double[][] gradWeight) {
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[i].length; j++) {

                // 1. 动量累计
                // 历史梯度方向
                mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * gradWeight[i][j];
                // 历史梯度幅值
                vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * gradWeight[i][j] * gradWeight[i][j];

                // 2. 偏差修正
                double mHat = mW[i][j] / (1 - Math.pow(beta1, t));
                double vHat = vW[i][j] / (1 - Math.pow(beta2, t));

                // 3. 应用
                weight[i][j] -= learningRate * mHat / Math.sqrt(vHat + eps);
            }

        }
    }

    @Override
    public void applyGradients(double[] bias, double[] gradBias) {
        for (int i = 0; i < bias.length; i++) {

            // -------- 更新偏置 --------
            mB[i] = beta1 * mB[i] + (1 - beta1) * gradBias[i];
            vB[i] = beta2 * vB[i] + (1 - beta2) * gradBias[i] * gradBias[i];

            double mHatB = mB[i] / (1 - Math.pow(beta1, t));
            double vHatB = vB[i] / (1 - Math.pow(beta2, t));

            bias[i] -= learningRate * mHatB / Math.sqrt(vHatB + eps);
        }
    }
}
