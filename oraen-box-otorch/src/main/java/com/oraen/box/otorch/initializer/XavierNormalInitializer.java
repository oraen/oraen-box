package com.oraen.box.otorch.initializer;

public class XavierNormalInitializer extends RandomInitializer{

    public static final XavierNormalInitializer INSTANCE = new XavierNormalInitializer();

    @Override
    double getNextRandom(int fanOut, int fanIn) {
        double limit = Math.sqrt(2.0 / (fanIn + fanOut));
        return random.nextGaussian() * limit;
    }
}
