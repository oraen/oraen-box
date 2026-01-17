package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import lombok.Getter;

@Getter
public class LayerNorm implements Layer<double[], double[]>, Learnable {

    private final int dim;
    private final double eps = 1e-5;

    // 可学习参数
    private final double[] gamma;
    private final double[] beta;

    // 梯度
    private final double[] gradGamma;
    private final double[] gradBeta;

    // forward 缓存（用于 backward）
    private double[][] input;
    private double[][] xHat;
    private double[] mean;
    private double[] var;

    // optimizer（和你其他 Learnable 一致）
    private GradOptimizer optimizer;

    public LayerNorm(int dim, GradOptimizer optimizer) {
        this.dim = dim;
        this.optimizer = optimizer;

        this.gamma = new double[dim];
        this.beta = new double[dim];
        this.gradGamma = new double[dim];
        this.gradBeta = new double[dim];

        // 常见初始化：gamma = 1, beta = 0
        for (int i = 0; i < dim; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    @Override
    public double[][] forwardBatch(double[][] data) {
        int batchSize = data.length;

        this.input = data;
        this.xHat = new double[batchSize][dim];
        this.mean = new double[batchSize];
        this.var = new double[batchSize];

        double[][] output = new double[batchSize][dim];

        for (int b = 0; b < batchSize; b++) {
            // 1. mean
            double m = 0;
            for (int i = 0; i < dim; i++) {
                m += data[b][i];
            }
            m /= dim;
            mean[b] = m;

            // 2. variance
            double v = 0;
            for (int i = 0; i < dim; i++) {
                double diff = data[b][i] - m;
                v += diff * diff;
            }
            v /= dim;
            var[b] = v;

            double stdInv = 1.0 / Math.sqrt(v + eps);

            // 3. normalize + scale + shift
            for (int i = 0; i < dim; i++) {
                xHat[b][i] = (data[b][i] - m) * stdInv;
                output[b][i] = gamma[i] * xHat[b][i] + beta[i];
            }
        }

        return output;
    }

    @Override
    public double[][] backwardBatch(double[][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;

        double[][] gradInput = new double[batchSize][dim];

        // 清空参数梯度
        for (int i = 0; i < dim; i++) {
            gradGamma[i] = 0;
            gradBeta[i] = 0;
        }

        for (int b = 0; b < batchSize; b++) {
            double stdInv = 1.0 / Math.sqrt(var[b] + eps);

            // -------- 参数梯度 --------
            for (int i = 0; i < dim; i++) {
                gradGamma[i] += gradOutputBatch[b][i] * xHat[b][i];
                gradBeta[i] += gradOutputBatch[b][i];
            }

            // -------- 输入梯度 --------
            double sumDxHat = 0;
            double sumDxHatXHat = 0;

            double[] dxHat = new double[dim];
            for (int i = 0; i < dim; i++) {
                dxHat[i] = gradOutputBatch[b][i] * gamma[i];
                sumDxHat += dxHat[i];
                sumDxHatXHat += dxHat[i] * xHat[b][i];
            }

            for (int i = 0; i < dim; i++) {
                gradInput[b][i] = stdInv * (
                        dxHat[i]
                                - sumDxHat / dim
                                - xHat[b][i] * sumDxHatXHat / dim
                );
            }
        }

        return gradInput;
    }

    @Override
    public void updateParameters() {
        // gamma / beta 是向量参数
        optimizer.applyGradients(gamma, gradGamma);
        optimizer.applyGradients(beta, gradBeta);
    }
}
