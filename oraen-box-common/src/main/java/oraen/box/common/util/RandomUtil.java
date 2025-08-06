package oraen.box.common.util;

import java.util.concurrent.ThreadLocalRandom;

public class RandomUtil {

    public static boolean rate(double probability) {
        if (probability < 0) {
            probability = 0;
        }

        if(probability > 1){
            probability = 1;
        }

        return ThreadLocalRandom.current().nextDouble() < probability;
    }
}
