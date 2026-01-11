package com.oraen.box.otorch.component;

import com.oraen.box.otorch.*;
import lombok.Data;

/**
 * 卷积神经网络层 - 教学版本
 *
 * 输入形状: [batchSize, inChannels, height, width]
 * 输出形状: [batchSize, outChannels, outHeight, outWidth]
 *
 * 计算规则:
 * outHeight = (height + 2*padding - kernelHeight) / stride + 1
 * outWidth = (width + 2*padding - kernelWidth) / stride + 1
 */
@Data
public class ConvolutionLayer implements Layer<double[][][], double[][][]>, Learnable {

    // ============ 卷积参数 ============
    private final double[][][][] kernels;  // [outChannels][inChannels][kH][kW]
    private final double[] biases;         // [outChannels]

    // ============ 维度信息 ============
    private final int inChannels;     // 输入通道数
    private final int outChannels;    // 输出通道数
    private final int kernelHeight;   // 卷积核高度
    private final int kernelWidth;    // 卷积核宽度
    private final int stride;         // 步长 (教学重点: 控制下采样)
    private final int padding;        // 填充 (教学重点: 保持输出尺寸)

    // ============ 训练组件 ============
    private GradOptimizer gradOptimizer;
    private final ParamInitializer paramInitializer;

    // ============ 训练状态 ============
    private double[][][][] cachedInputBatch;  // 缓存的输入，用于反向传播
    private double[][][][] gradKernels;       // 卷积核梯度
    private double[] gradBiases;              // 偏置梯度
    private int gradientAccumulationCount;    // 梯度累积计数

    public ConvolutionLayer(int inChannels, int outChannels,
                            int kernelHeight, int kernelWidth,
                            int stride, int padding,
                            ParamInitializer paramInitializer,
                            GradOptimizer gradOptimizer) {

        // 参数校验
        if (stride <= 0) throw new IllegalArgumentException("步长必须大于0");
        if (padding < 0) throw new IllegalArgumentException("填充不能为负数");

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;      // 新增：步长参数
        this.padding = padding;    // 新增：填充参数
        this.paramInitializer = paramInitializer;
        this.gradOptimizer = gradOptimizer;

        // 初始化参数
        this.kernels = new double[outChannels][inChannels][kernelHeight][kernelWidth];
        this.biases = new double[outChannels];
        paramInitializer.initialize(kernels);
        paramInitializer.initializeBiases(biases);

        // 初始化梯度
        this.gradKernels = new double[outChannels][inChannels][kernelHeight][kernelWidth];
        this.gradBiases = new double[outChannels];
        this.gradientAccumulationCount = 0;
    }

    /**
     * 简化的构造器，步长=1，填充=0
     */
    public ConvolutionLayer(int inChannels, int outChannels,
                            int kernelHeight, int kernelWidth,
                            ParamInitializer paramInitializer,
                            GradOptimizer gradOptimizer) {
        this(inChannels, outChannels, kernelHeight, kernelWidth,
                1, 0, paramInitializer, gradOptimizer);
    }

    @Override
    public double[][][][] forwardBatch(double[][][][] inputBatch) {
        int batchSize = inputBatch.length;
        double[][][][] outputBatch = new double[batchSize][][][];

        // 缓存输入用于反向传播
        this.cachedInputBatch = inputBatch;

        for (int b = 0; b < batchSize; b++) {
            // 步骤1: 添加填充
            double[][][] paddedInput = addPadding(inputBatch[b]);

            // 步骤2: 计算输出尺寸
            int outHeight = calculateOutputSize(paddedInput[0].length, kernelHeight, stride);
            int outWidth = calculateOutputSize(paddedInput[0][0].length, kernelWidth, stride);

            // 步骤3: 执行卷积
            outputBatch[b] = new double[outChannels][outHeight][outWidth];

            // 对每个输出通道计算卷积
            for (int oc = 0; oc < outChannels; oc++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        // 教学重点: 使用步长计算输入起始位置
                        int inputStartH = oh * stride;
                        int inputStartW = ow * stride;

                        outputBatch[b][oc][oh][ow] = biases[oc] +
                                computeConvolutionAtPosition(paddedInput, kernels[oc],
                                        inputStartH, inputStartW);
                    }
                }
            }
        }

        return outputBatch;
    }

    /**
     * 为输入添加填充
     * 教学重点: 填充的作用是保持输出尺寸
     */
    private double[][][] addPadding(double[][][] input) {
        if (padding == 0) {
            return input;  // 无填充，直接返回
        }

        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;

        int paddedHeight = height + 2 * padding;
        int paddedWidth = width + 2 * padding;

        double[][][] padded = new double[channels][paddedHeight][paddedWidth];

        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < paddedHeight; h++) {
                for (int w = 0; w < paddedWidth; w++) {
                    int origH = h - padding;
                    int origW = w - padding;

                    if (origH >= 0 && origH < height && origW >= 0 && origW < width) {
                        padded[c][h][w] = input[c][origH][origW];
                    } else {
                        padded[c][h][w] = 0.0;  // 零填充
                    }
                }
            }
        }

        return padded;
    }

    /**
     * 计算输出尺寸
     * 教学重点: 理解卷积输出尺寸公式
     */
    private int calculateOutputSize(int inputSize, int kernelSize, int stride) {
        // 公式: (inputSize + 2*padding - kernelSize) / stride + 1
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
    }

    /**
     * 在指定位置计算卷积值
     */
    private double computeConvolutionAtPosition(double[][][] input,
                                                double[][][] kernelsForChannel,
                                                int startRow, int startCol) {
        double sum = 0.0;

        // 遍历所有输入通道
        for (int ic = 0; ic < inChannels; ic++) {
            // 遍历卷积核的每个位置
            for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                    int inputRow = startRow + kh;
                    int inputCol = startCol + kw;

                    double inputValue = input[ic][inputRow][inputCol];
                    double weight = kernelsForChannel[ic][kh][kw];

                    sum += inputValue * weight;
                }
            }
        }

        return sum;
    }

    @Override
    public double[][][][] backwardBatch(double[][][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;
        double[][][][] gradInputBatch = new double[batchSize][inChannels][][];

        // 累积梯度计数
        gradientAccumulationCount += batchSize;

        for (int b = 0; b < batchSize; b++) {
            // 步骤1: 为输入添加填充（和forward时一致）
            double[][][] paddedInput = addPadding(cachedInputBatch[b]);

            // 步骤2: 计算反向传播
            gradInputBatch[b] = backward0(gradOutputBatch[b], paddedInput);
        }

        return gradInputBatch;
    }

    private double[][][] backward0(double[][][] gradOutput, double[][][] paddedInput) {
        int paddedHeight = paddedInput[0].length;
        int paddedWidth = paddedInput[0][0].length;
        int outHeight = gradOutput[0].length;
        int outWidth = gradOutput[0][0].length;

        // 初始化输入梯度（带填充的）
        double[][][] gradPaddedInput = new double[inChannels][paddedHeight][paddedWidth];

        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outHeight; oh++) {
                for (int ow = 0; ow < outWidth; ow++) {
                    double grad = gradOutput[oc][oh][ow];

                    // 教学重点: 反向传播时也要考虑步长
                    int inputStartH = oh * stride;
                    int inputStartW = ow * stride;

                    // 累加偏置梯度
                    gradBiases[oc] += grad;

                    // 计算当前位置的梯度贡献
                    computeGradientAtPosition(grad, paddedInput, gradPaddedInput,
                            kernels[oc], oc, inputStartH, inputStartW);
                }
            }
        }

        // 移除填充，得到真实的输入梯度
        return removePadding(gradPaddedInput);
    }

    /**
     * 从带填充的梯度中移除填充
     */
    private double[][][] removePadding(double[][][] gradPaddedInput) {
        if (padding == 0) {
            return gradPaddedInput;
        }

        int channels = gradPaddedInput.length;
        int paddedHeight = gradPaddedInput[0].length;
        int paddedWidth = gradPaddedInput[0][0].length;

        int height = paddedHeight - 2 * padding;
        int width = paddedWidth - 2 * padding;

        double[][][] gradInput = new double[channels][height][width];

        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int paddedH = h + padding;
                    int paddedW = w + padding;
                    gradInput[c][h][w] = gradPaddedInput[c][paddedH][paddedW];
                }
            }
        }

        return gradInput;
    }

    private void computeGradientAtPosition(double gradValue,
                                           double[][][] input,
                                           double[][][] gradInput,
                                           double[][][] kernelsForChannel,
                                           int outputChannel,
                                           int startRow, int startCol) {
        for (int ic = 0; ic < inChannels; ic++) {
            for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                    int inputRow = startRow + kh;
                    int inputCol = startCol + kw;

                    // 卷积核梯度: ∂L/∂W = ∂L/∂O * X
                    gradKernels[outputChannel][ic][kh][kw] +=
                            gradValue * input[ic][inputRow][inputCol];

                    // 输入梯度: ∂L/∂X = ∂L/∂O * W
                    gradInput[ic][inputRow][inputCol] +=
                            gradValue * kernelsForChannel[ic][kh][kw];
                }
            }
        }
    }

    @Override
    public void updateParameters() {
        if (gradientAccumulationCount > 0) {
            // 平均梯度
            double scale = 1.0 / gradientAccumulationCount;
            scaleGradients(scale);
        }

        // 应用梯度更新
        applyParameterUpdates();

        // 重置状态
        clearGradients();
        gradientAccumulationCount = 0;
    }

    private void scaleGradients(double scale) {
        for (int oc = 0; oc < outChannels; oc++) {
            // 缩放偏置梯度
            gradBiases[oc] *= scale;

            for (int ic = 0; ic < inChannels; ic++) {
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        gradKernels[oc][ic][kh][kw] *= scale;
                    }
                }
            }
        }
    }

    private void applyParameterUpdates() {
        // 更新卷积核
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                gradOptimizer.applyGradients(kernels[oc][ic], gradKernels[oc][ic]);
            }
        }

        // 更新偏置
        gradOptimizer.applyGradients(biases, gradBiases);
    }

    private void clearGradients() {
        // 清零卷积核梯度
        for (int oc = 0; oc < outChannels; oc++) {
            gradBiases[oc] = 0.0;
            for (int ic = 0; ic < inChannels; ic++) {
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        gradKernels[oc][ic][kh][kw] = 0.0;
                    }
                }
            }
        }
    }

    // ============ 教学辅助方法 ============

    /**
     * 获取卷积层的参数数量（用于计算复杂度）
     */
    public int getParameterCount() {
        int weightCount = outChannels * inChannels * kernelHeight * kernelWidth;
        int biasCount = outChannels;
        return weightCount + biasCount;
    }

    /**
     * 获取输出特征图尺寸
     */
    public int[] getOutputShape(int inputHeight, int inputWidth) {
        int outHeight = calculateOutputSize(inputHeight + 2 * padding, kernelHeight, stride);
        int outWidth = calculateOutputSize(inputWidth + 2 * padding, kernelWidth, stride);
        return new int[]{outChannels, outHeight, outWidth};
    }

    /**
     * 打印层信息（用于教学演示）
     */
    public void printLayerInfo(int inputHeight, int inputWidth) {
        System.out.println("=== 卷积层信息 ===");
        System.out.println("输入通道: " + inChannels);
        System.out.println("输出通道: " + outChannels);
        System.out.println("卷积核尺寸: " + kernelHeight + "×" + kernelWidth);
        System.out.println("步长: " + stride);
        System.out.println("填充: " + padding);

        int[] outputShape = getOutputShape(inputHeight, inputWidth);
        System.out.println("输出尺寸: " + outputShape[0] + "×" +
                outputShape[1] + "×" + outputShape[2]);

        System.out.println("参数数量: " + getParameterCount());
        System.out.println("==================");
    }
}