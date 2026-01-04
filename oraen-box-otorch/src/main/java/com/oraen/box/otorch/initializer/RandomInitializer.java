package com.oraen.box.otorch.initializer;

import com.oraen.box.otorch.ParamInitializer;
import lombok.Data;

import java.util.Random;

@Data
public abstract class RandomInitializer implements ParamInitializer {

    protected Random random;

    private double biasValue;

    public RandomInitializer() {
        this(0.0);
    }

    public RandomInitializer(Random random) {
        this(random, 0.0);
    }

    public RandomInitializer(double biasValue) {
        this(new Random(), biasValue);
    }

    public RandomInitializer(Random random, double biasValue) {
        this.random = random;
        this.biasValue = biasValue;
    }

    abstract double getNextRandom(int fanOut, int fanIn);



    @Override
    public void initializeWeights(double[][] weights) {
        int fanOut = weights.length;
        int fanIn = weights[0].length;

        for (int i = 0; i < fanOut; i++) {
            for (int j = 0; j < fanIn; j++) {
                weights[i][j] = getNextRandom(fanOut, fanIn);
            }
        }

    }

    @Override
    public void initializeBiases(double[] biases) {
        int fanOut = biases.length;
        // Xavier初始化通常将偏置设为0
        for (int i = 0; i < fanOut; i++) {
            biases[i] = biasValue;
        }

    }
}
