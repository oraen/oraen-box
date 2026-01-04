package com.oraen.box.otorch.initializer;

import com.oraen.box.otorch.ParamInitializer;
import lombok.Data;

/**
 * 常量初始化器（主要用于测试和调试）
 */
@Data
public class ConstantInitializer extends RandomInitializer {

    double weightValue;
    
    public ConstantInitializer(double weightValue, double biasValue) {
        super(biasValue);
        this.weightValue = weightValue;
    }

    @Override
    double getNextRandom(int fanOut, int fanIn) {
        return weightValue;
    }

}