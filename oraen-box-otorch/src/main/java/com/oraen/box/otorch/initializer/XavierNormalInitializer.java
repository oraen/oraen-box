package com.oraen.box.otorch.initializer;

public class XavierNormalInitializer extends RandomInitializer{
    @Override
    double getNextRandom(int fanOut, int fanIn) {
        double limit = Math.sqrt(2.0 / (fanIn + fanOut));
        return random.nextGaussian() * limit;
    }
}
