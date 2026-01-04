package com.oraen.box.common.util;

public class MathUtil {

    public static double sum(double... nums){
        double re = 0;
        for(double num : nums){
            re += num;
        }
        return re;
    }
}
