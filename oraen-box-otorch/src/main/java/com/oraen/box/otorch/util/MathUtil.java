package com.oraen.box.otorch.util;

public class MathUtil {

    public static double mean(double[] arr) {
        double sum = 0.0;
        for (double v : arr) {
            sum += v;
        }
        return sum / arr.length;
    }

    public static double variance(double[] arr, double mean) {
        double sum = 0.0;
        for (double v : arr) {
            sum += (v - mean) * (v - mean);
        }
        return sum / arr.length;
    }

    public static double variance(double[] arr) {
        double mean = mean(arr);
        return variance(arr, mean);
    }
}
