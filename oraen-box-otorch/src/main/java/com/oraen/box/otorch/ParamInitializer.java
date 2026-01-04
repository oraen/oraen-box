package com.oraen.box.otorch;

public interface ParamInitializer {

    /**
     * 直接初始化给定的权重矩阵
     * @param weights 需要初始化的权重矩阵
     */
    void initializeWeights(double[][] weights);

    /**
     * 直接初始化给定的偏置向量
     * @param biases 需要初始化的偏置向量
     */
    void initializeBiases(double[] biases);
}
