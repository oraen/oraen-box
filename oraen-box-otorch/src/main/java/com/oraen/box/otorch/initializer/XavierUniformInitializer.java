package com.oraen.box.otorch.initializer;

public class XavierUniformInitializer extends RandomInitializer{

    public static final XavierUniformInitializer INSTANCE = new XavierUniformInitializer();

    @Override
    double getNextRandom(int fanOut, int fanIn) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        return random.nextDouble() * 2 * limit - limit;
    }
}
