package com.oraen.box.otorch.util;

import java.util.Arrays;

public class DataUtil {

    public static double[] copy(double[] ds){
        double[] copy = new double[ds.length];
        System.arraycopy(ds, 0, copy, 0, ds.length);
        return copy;
    }

    public static double[][] copy(double[][] ds){
        double[][] copy = new double[ds.length][];
        for(int i = 0; i < ds.length; i++){
            copy[i] = copy(ds[i]);
        }
        return copy;
    }

    public static double[][][] copy(double[][][] ds){
        double[][][] copy = new double[ds.length][][];
        for(int i = 0; i < ds.length; i++){
            copy[i] = copy(ds[i]);
        }
        return copy;
    }

    public static void clear(double[][]... ds){
        for(double [][] d : ds){
            for(double[] row : d){
                Arrays.fill(row, 0);
            }
        }
    }

    public static void paintData(double[][] source, double[][] target, int offsetRow, int offsetCol){
        for(int i = 0; i < source.length; i++){
            System.arraycopy(source[i], 0, target[offsetRow + i], offsetCol, source[i].length);
        }
    }
}
