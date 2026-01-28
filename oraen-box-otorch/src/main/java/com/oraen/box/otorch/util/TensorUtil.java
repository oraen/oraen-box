package com.oraen.box.otorch.util;

public class TensorUtil {

    /**
     * 矩阵乘法
     */
    public static double[][] matmul(double[][] a, double[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Incompatible dimensions: A columns (" + a[0].length + ") != B rows (" + b.length + ")");
        }

        int m = a.length;        // rows of A
        int n = a[0].length;     // cols of A == rows of B
        int p = b[0].length;     // cols of B

        double[][] c = new double[m][p];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }

        return c;
    }

    public static double[][] project(double[][] a, double[][] b) {
        int rowNum = a.length;
        int colNum = b.length;
        int dim = b[0].length;
        double[][] out = new double[rowNum][colNum];
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                double val = 0;
                for (int k = 0; k < dim; k++) {
                    val += a[i][k] * b[j][k];
                }
                out[i][j] = val;
            }
        }

        return out;
    }


    /**
     * 矩阵转置
     *
     * @param matrix 输入矩阵，shape [M, N]
     * @return 转置矩阵，shape [N, M]
     */
    public static double[][] transpose(double[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new double[0][0];
        }

        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    public static double[] add(double[] a, double[] b) {
        int length = a.length;
        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }



    public static double[][] add(double[][] a, double[][] b) {
        double[][] result = new double[a.length][];
        for(int i=0; i<a.length; i++) {
            result[i] = add(a[i], b[i]);
        }

        return result;
    }

    public static double[][][] add(double[][][] a, double[][][] b) {
        double[][][] result = new double[a.length][][];
        for(int i=0; i<a.length; i++) {
            result[i] = add(a[i], b[i]);
        }

        return result;
    }

}



