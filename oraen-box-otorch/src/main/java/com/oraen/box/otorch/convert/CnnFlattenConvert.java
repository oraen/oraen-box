package com.oraen.box.otorch.convert;

import com.oraen.box.otorch.Layer;

public class CnnFlattenConvert  implements Layer<double[][][], double[]> {
    // 缓存原始形状用于反向传播
    private int[] originalShape;

    private double[] forward0(double[][][] input) {
        // 保存原始形状
        originalShape = new int[]{
                input.length,      // channels
                input[0].length,   // height
                input[0][0].length // width
        };

        // 计算总元素数
        int totalElements = input.length * input[0].length * input[0][0].length;
        double[] output = new double[totalElements];

        // 展平操作
        int index = 0;
        for (int c = 0; c < input.length; c++) {
            for (int h = 0; h < input[0].length; h++) {
                for (int w = 0; w < input[0][0].length; w++) {
                    output[index++] = input[c][h][w];
                }
            }
        }

        return output;
    }


    private double[][][] backward0(double[] gradOutput) {
        // 重塑梯度到原始形状
        double[][][] gradInput = new double[originalShape[0]][originalShape[1]][originalShape[2]];

        int index = 0;
        for (int c = 0; c < originalShape[0]; c++) {
            for (int h = 0; h < originalShape[1]; h++) {
                for (int w = 0; w < originalShape[2]; w++) {
                    gradInput[c][h][w] = gradOutput[index++];
                }
            }
        }

        return gradInput;
    }

    @Override
    public double[][] forwardBatch(double[][][][] data) {
        double[][] re = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            re[i] = forward0(data[i]);
        }
        return re;
    }

    @Override
    public double[][][][] backwardBatch(double[][] gradOutputBatch) {
        double[][][][] re = new double[gradOutputBatch.length][][][];;
        for(int i = 0; i < gradOutputBatch.length; i++) {
            re[i] = backward0(gradOutputBatch[i]);
        }
        return re;
    }
}
