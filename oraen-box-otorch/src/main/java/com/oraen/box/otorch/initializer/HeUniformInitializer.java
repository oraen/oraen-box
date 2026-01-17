package com.oraen.box.otorch.initializer;

public class HeUniformInitializer extends RandomInitializer{

    public static final HeUniformInitializer INSTANCE = new HeUniformInitializer();

    public HeUniformInitializer(){
        super(0.01);
    }

    @Override
    double getNextRandom(int fanOut, int fanIn) {
        double limit = Math.sqrt(6.0 / fanIn);
        return random.nextDouble() * 2 * limit - limit;
    }
}
