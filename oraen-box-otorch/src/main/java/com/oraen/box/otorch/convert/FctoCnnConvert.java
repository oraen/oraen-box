package com.oraen.box.otorch.convert;

import com.oraen.box.otorch.Layer;

public class FctoCnnConvert implements Layer<double[], double[][][]> {
    
    // 目标形状 [channels, height, width]
    private final int[] targetShape;
    
    public FctoCnnConvert(int channels, int height, int width) {
        this.targetShape = new int[]{channels, height, width};
        
        // 验证参数
        if (channels <= 0 || height <= 0 || width <= 0) {
            throw new IllegalArgumentException("目标形状参数必须为正数");
        }
    }
    
    public FctoCnnConvert(int[] shape) {
        if (shape.length != 3) {
            throw new IllegalArgumentException("目标形状必须是3维数组");
        }
        this.targetShape = shape.clone();
    }
    
    private double[][][] forward0(double[] input) {
        // 验证输入长度
        int expectedLength = targetShape[0] * targetShape[1] * targetShape[2];
        if (input.length != expectedLength) {
            throw new IllegalArgumentException(
                String.format("输入长度不匹配。期望: %d (C=%d,H=%d,W=%d)，实际: %d",
                    expectedLength, targetShape[0], targetShape[1], targetShape[2], 
                    input.length));
        }
        
        // 重塑为3D数组
        double[][][] output = new double[targetShape[0]][targetShape[1]][targetShape[2]];
        
        int index = 0;
        for (int c = 0; c < targetShape[0]; c++) {
            for (int h = 0; h < targetShape[1]; h++) {
                for (int w = 0; w < targetShape[2]; w++) {
                    output[c][h][w] = input[index++];
                }
            }
        }
        
        return output;
    }
    
    private double[] backward0(double[][][] gradOutput) {
        // 验证输入形状
        if (gradOutput.length != targetShape[0] ||
            gradOutput[0].length != targetShape[1] ||
            gradOutput[0][0].length != targetShape[2]) {
            throw new IllegalArgumentException(
                String.format("梯度形状不匹配。期望: [%d][%d][%d]，实际: [%d][%d][%d]",
                    targetShape[0], targetShape[1], targetShape[2],
                    gradOutput.length, gradOutput[0].length, gradOutput[0][0].length));
        }
        
        // 展平为1D数组
        int totalElements = targetShape[0] * targetShape[1] * targetShape[2];
        double[] gradInput = new double[totalElements];
        
        int index = 0;
        for (int c = 0; c < targetShape[0]; c++) {
            for (int h = 0; h < targetShape[1]; h++) {
                for (int w = 0; w < targetShape[2]; w++) {
                    gradInput[index++] = gradOutput[c][h][w];
                }
            }
        }
        
        return gradInput;
    }
    
    @Override
    public double[][][][] forwardBatch(double[][] data) {
        double[][][][] output = new double[data.length][][][];
        
        for (int i = 0; i < data.length; i++) {
            output[i] = forward0(data[i]);
        }
        
        return output;
    }
    
    @Override
    public double[][] backwardBatch(double[][][][] gradOutputBatch) {
        double[][] gradInputBatch = new double[gradOutputBatch.length][];
        
        for (int i = 0; i < gradOutputBatch.length; i++) {
            gradInputBatch[i] = backward0(gradOutputBatch[i]);
        }
        
        return gradInputBatch;
    }
    

}