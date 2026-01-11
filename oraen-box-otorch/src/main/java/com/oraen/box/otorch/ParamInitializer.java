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


    /**
     * 通用方法
     */
    /**
     * 通用方法：初始化任意维度的double数组
     * 支持 double[], double[][], double[][][] 等
     */
    default void initialize(Object array) {
        if (array == null) {
            return;
        }

        if (array instanceof double[]) {
            initializeBiases((double[]) array);
            // double[] 不instanceof Object[] 但是double[][] instanceof Object[], 因为double[]是对象
        } else if (array instanceof Object[]) {
            // 处理更高维度的数组
            for (Object element : (Object[]) array) {
                if (element != null) {
                    initialize(element);
                }
            }
        }
    }
}
