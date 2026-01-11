package com.oraen.box.otorch.component;

import com.oraen.box.otorch.Layer;
import lombok.Data;

/**
 * 池化神经网络层 - 教学版本
 *
 * 输入形状: [batchSize, channels, height, width]
 * 输出形状: [batchSize, channels, outHeight, outWidth]
 *
 * 计算规则:
 * outHeight = (height - poolHeight) / stride + 1
 * outWidth = (width - poolWidth) / stride + 1
 *
 * 注意: 池化层不改变通道数，只改变空间尺寸
 */
@Data
public class PoolLayer implements Layer<double[][][], double[][][]> {

    public enum PoolType {
        MAX,     // 最大池化
        AVG      // 平均池化
    }

    // ============ 池化参数 ============
    private final int poolHeight;   // 池化窗口高度
    private final int poolWidth;    // 池化窗口宽度
    private final int stride;       // 步长
    private final PoolType poolType; // 池化类型

    // ============ 训练状态 ============
    private double[][][][] cachedInputBatch;  // 缓存的输入，用于反向传播
    private int[][][][][] maxIndices;         // 最大池化的索引缓存 [batch][channel][outH][outW][2]

    public PoolLayer(int poolHeight, int poolWidth, int stride, PoolType poolType) {
        // 参数校验
        if (poolHeight <= 0 || poolWidth <= 0) {
            throw new IllegalArgumentException("池化窗口尺寸必须大于0");
        }
        if (stride <= 0) {
            throw new IllegalArgumentException("步长必须大于0");
        }

        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
        this.stride = stride;
        this.poolType = poolType;
    }

    /**
     * 简化的构造器，步长等于池化窗口尺寸（不重叠池化）
     */
    public PoolLayer(int poolSize, PoolType poolType) {
        this(poolSize, poolSize, poolSize, poolType);
    }

    @Override
    public double[][][][] forwardBatch(double[][][][] inputBatch) {
        int batchSize = inputBatch.length;

        // 验证输入维度
        validateInputDimensions(inputBatch[0]);

        // 缓存输入用于反向传播
        this.cachedInputBatch = inputBatch;

        // 计算输出尺寸
        int channels = inputBatch[0].length;
        int height = inputBatch[0][0].length;
        int width = inputBatch[0][0][0].length;

        int outHeight = calculateOutputSize(height, poolHeight, stride);
        int outWidth = calculateOutputSize(width, poolWidth, stride);

        // 初始化输出和索引缓存
        double[][][][] outputBatch = new double[batchSize][channels][outHeight][outWidth];

        if (poolType == PoolType.MAX) {
            // 只在最大池化时初始化索引缓存
            this.maxIndices = new int[batchSize][channels][outHeight][outWidth][2];
        }

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                // 对每个通道独立池化
                outputBatch[b][c] = pool2D(inputBatch[b][c], b, c, outHeight, outWidth);
            }
        }

        return outputBatch;
    }

    /**
     * 二维池化操作
     */
    private double[][] pool2D(double[][] input, int batchIdx, int channelIdx,
                              int outHeight, int outWidth) {
        double[][] output = new double[outHeight][outWidth];

        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                // 计算输入窗口起始位置
                int startH = oh * stride;
                int startW = ow * stride;

                // 执行池化
                if (poolType == PoolType.MAX) {
                    output[oh][ow] = maxPool(input, startH, startW, batchIdx, channelIdx, oh, ow);
                } else {
                    output[oh][ow] = avgPool(input, startH, startW);
                }
            }
        }

        return output;
    }

    /**
     * 最大池化
     */
    private double maxPool(double[][] input, int startH, int startW,
                           int batchIdx, int channelIdx, int oh, int ow) {
        double maxVal = -Double.MAX_VALUE;
        int maxH = -1, maxW = -1;

        // 遍历池化窗口
        for (int ph = 0; ph < poolHeight; ph++) {
            for (int pw = 0; pw < poolWidth; pw++) {
                int h = startH + ph;
                int w = startW + pw;

                if (h < input.length && w < input[0].length) {
                    double val = input[h][w];
                    if (val > maxVal) {
                        maxVal = val;
                        maxH = h;
                        maxW = w;
                    }
                }
            }
        }

        // 缓存最大值的位置（用于反向传播）
        if (maxH != -1 && maxW != -1) {
            maxIndices[batchIdx][channelIdx][oh][ow][0] = maxH;
            maxIndices[batchIdx][channelIdx][oh][ow][1] = maxW;
        }

        return maxVal;
    }

    /**
     * 平均池化
     */
    private double avgPool(double[][] input, int startH, int startW) {
        double sum = 0.0;
        int count = 0;

        // 遍历池化窗口
        for (int ph = 0; ph < poolHeight; ph++) {
            for (int pw = 0; pw < poolWidth; pw++) {
                int h = startH + ph;
                int w = startW + pw;

                if (h < input.length && w < input[0].length) {
                    sum += input[h][w];
                    count++;
                }
            }
        }

        return count > 0 ? sum / count : 0.0;
    }

    @Override
    public double[][][][] backwardBatch(double[][][][] gradOutputBatch) {
        int batchSize = gradOutputBatch.length;

        // 验证梯度维度
        if (cachedInputBatch == null) {
            throw new IllegalStateException("必须先调用forwardBatch才能进行反向传播");
        }

        int channels = cachedInputBatch[0].length;
        int height = cachedInputBatch[0][0].length;
        int width = cachedInputBatch[0][0][0].length;

        // 初始化输入梯度
        double[][][][] gradInputBatch = new double[batchSize][channels][height][width];

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                gradInputBatch[b][c] = poolBackward2D(
                        gradOutputBatch[b][c], b, c, height, width);
            }
        }

        return gradInputBatch;
    }

    /**
     * 二维池化的反向传播
     */
    private double[][] poolBackward2D(double[][] gradOutput,
                                      int batchIdx, int channelIdx,
                                      int inputHeight, int inputWidth) {
        double[][] gradInput = new double[inputHeight][inputWidth];
        int outHeight = gradOutput.length;
        int outWidth = gradOutput[0].length;

        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                double grad = gradOutput[oh][ow];
                int startH = oh * stride;
                int startW = ow * stride;

                if (poolType == PoolType.MAX) {
                    // 最大池化：梯度只传递到最大值的位置
                    int maxH = maxIndices[batchIdx][channelIdx][oh][ow][0];
                    int maxW = maxIndices[batchIdx][channelIdx][oh][ow][1];

                    if (maxH != -1 && maxW != -1) {
                        gradInput[maxH][maxW] += grad;
                    }
                } else {
                    // 平均池化：梯度平均分配到窗口内所有位置
                    distributeAvgGradient(gradInput, grad, startH, startW);
                }
            }
        }

        return gradInput;
    }

    /**
     * 分布平均池化的梯度
     */
    private void distributeAvgGradient(double[][] gradInput, double grad,
                                       int startH, int startW) {
        int count = 0;

        // 先计算有效位置数量
        for (int ph = 0; ph < poolHeight; ph++) {
            for (int pw = 0; pw < poolWidth; pw++) {
                int h = startH + ph;
                int w = startW + pw;

                if (h < gradInput.length && w < gradInput[0].length) {
                    count++;
                }
            }
        }

        if (count > 0) {
            double gradPerCell = grad / count;

            // 分配梯度
            for (int ph = 0; ph < poolHeight; ph++) {
                for (int pw = 0; pw < poolWidth; pw++) {
                    int h = startH + ph;
                    int w = startW + pw;

                    if (h < gradInput.length && w < gradInput[0].length) {
                        gradInput[h][w] += gradPerCell;
                    }
                }
            }
        }
    }

    /**
     * 计算输出尺寸
     */
    private int calculateOutputSize(int inputSize, int poolSize, int stride) {
        return (inputSize - poolSize) / stride + 1;
    }

    /**
     * 验证输入维度
     */
    private void validateInputDimensions(double[][][] input) {
        if (input == null || input.length == 0 || input[0].length == 0 || input[0][0].length == 0) {
            throw new IllegalArgumentException("输入维度无效");
        }

        int height = input[0].length;
        int width = input[0][0].length;

        // 检查输入尺寸是否足够
        if (height < poolHeight || width < poolWidth) {
            throw new IllegalArgumentException(
                    String.format("输入尺寸太小: %d×%d < 池化窗口: %d×%d",
                            height, width, poolHeight, poolWidth));
        }
    }



}