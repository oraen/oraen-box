package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.GradOptimizer;
import com.oraen.box.otorch.Layer;
import com.oraen.box.otorch.Learnable;
import com.oraen.box.otorch.ParamInitializer;
import com.oraen.box.otorch.util.MathUtil;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

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
    private double[][] xHat;
    private double[] var;

    @Setter
    // optimizer（和你其他 Learnable 一致）
    private GradOptimizer optimizer;

    /**
     *
     * @param dim 维度
     * @param optimizer 参数优化器
     * @param gammaParamInitializer 参数初始化器，一般是使用ConstantInitializer， gamma都初始化为1，
     * @param betaParamInitializer 参数初始化器，一般是使用ConstantInitializer，  beta都初始化为0
     */
    public LayerNorm(int dim, GradOptimizer optimizer, ParamInitializer gammaParamInitializer, ParamInitializer betaParamInitializer) {
        this.dim = dim;
        this.optimizer = optimizer;

        this.gamma = new double[dim];
        this.beta = new double[dim];
        this.gradGamma = new double[dim];
        this.gradBeta = new double[dim];

        gammaParamInitializer.initializeBiases(this.gamma);
        betaParamInitializer.initializeBiases(this.beta);
    }

    @Override
    public double[][] forwardBatch(double[][] data) {
        // batchSize：当前 batch 中样本数量（batch 维度）
        int batchSize = data.length;

        // xHat：归一化后的中间结果 (x - mean) / sqrt(var + eps)
        this.xHat = new double[batchSize][];

        // var：每个样本在 feature 维度上的方差
        this.var = new double[batchSize];

        // output：最终前向输出
        double[][] output = new double[batchSize][dim];

        for (int b = 0; b < batchSize; b++) {
            output[b] = forward0(data[b], b);
        }

        return output;
    }

    @Override
    public double[][] backwardBatch(double[][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        double[][] gradInput = new double[batchSize][dim];
        for (int b = 0; b < batchSize; b++) {
            gradInput[b] = backward0(gradOutputBatch[b], b);
        }
        return gradInput;
    }

    @Override
    public void updateParameters() {
        // gamma / beta 是向量参数
        optimizer.applyGradients(gamma, gradGamma);
        optimizer.applyGradients(beta, gradBeta);
        clearGrad();
    }

    private void clearGrad(){
        Arrays.fill(gradGamma, 0);
        Arrays.fill(gradBeta, 0);
    }


    public double[] forward0(double[] data, int batchIndex) {

        // 输出
        double[] output = new double[dim];
        //平均值
        double mean = MathUtil.mean(data);
        //方差
        double var = MathUtil.variance(data, mean);
        // std：标准差，用于数值稳定，eps 防止除0
        double std = Math.sqrt(var + eps);

        //归一化（normalize）+ 缩放（gamma）+ 平移（beta）
        double[] xHat = new double[dim];
        for (int i = 0; i < dim; i++) {
            xHat[i] = (data[i] - mean) / std;
            // output = gamma * xHat + beta
            output[i] = gamma[i] * xHat[i] + beta[i];
        }

        //缓存xHat
        this.xHat[batchIndex] = xHat;
        this.var[batchIndex] = var;

        return output;
    }


    private double[] backward0(double[] gradOutput, int batchIndex) {
        // 更新参数梯度
        updateGrad(gradOutput, batchIndex);

        double[] gradInput = new double[dim];
        double std = Math.sqrt(var[batchIndex] + eps);
        double sumDxHat = 0;
        double sumDxHatXHat = 0;

        double[] dxHat = new double[dim];
        for (int i = 0; i < dim; i++) {
            dxHat[i] = gradOutput[i] * gamma[i];
            sumDxHat += dxHat[i];
            sumDxHatXHat += dxHat[i] * xHat[batchIndex][i];
        }

        for(int i = 0; i < dim; i ++){
            gradInput[i] = (dxHat[i] - sumDxHat / dim - xHat[batchIndex][i] * sumDxHatXHat / dim) / std;
        }

        return gradInput;
    }

    // 更新参数梯度
    private void updateGrad(double[] gradOutput, int batchIndex) {
        for (int i = 0; i < dim; i++) {
            gradGamma[i] += gradOutput[i] * xHat[batchIndex][i];
            gradBeta[i] += gradOutput[i];
        }
    }
}
