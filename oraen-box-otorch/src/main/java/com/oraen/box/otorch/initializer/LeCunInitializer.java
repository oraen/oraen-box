package com.oraen.box.otorch.initializer;

import com.oraen.box.otorch.ParamInitializer;

import java.util.Random;

/**
 * LeCun 初始化器（适用于 SELU/Tanh）
 * 论文: Efficient BackProp
 */
public class LeCunInitializer extends RandomInitializer{

    @Override
    double getNextRandom(int fanOut, int fanIn) {
        double stdDev = Math.sqrt(1.0 / fanIn);
        return random.nextGaussian() * stdDev;
    }


}