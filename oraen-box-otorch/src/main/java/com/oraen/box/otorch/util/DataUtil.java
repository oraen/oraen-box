package com.oraen.box.otorch.util;

public class DataUtil {

    public static double[] copy(double[] ds){
        double[] copy = new double[ds.length];
        System.arraycopy(ds, 0, copy, 0, ds.length);
        return copy;
    }
}
