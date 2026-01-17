package com.oraen.box.otorch.initializer;

public class HeNormalInitializer extends RandomInitializer{

    public static final HeNormalInitializer INSTANCE = new HeNormalInitializer();

    public HeNormalInitializer(){
        super(0.01);
    }

    @Override
    double getNextRandom(int fanOut, int fanIn) {
        // 正态分布 N(0, 2/fan_in)
        double stdDev = Math.sqrt(2.0 / fanIn);

        return random.nextGaussian() * stdDev;
    }
}
