package com.oraen.box.otorch.convert;

import com.oraen.box.otorch.Layer;

/**
 * Merge first two dims: [a][b][c] -> [a*b][c]
 */
class MergeFirstDimsConvert implements Layer<double[][], double[]> {


    private final int firstDim;
    private final int secondDim;
    private final int thirdDim;


    public MergeFirstDimsConvert(int firstDim, int secondDim, int thirdDim) {
        this.firstDim = firstDim;
        this.secondDim = secondDim;
        this.thirdDim = thirdDim;
    }


    @Override
    public double[][] forwardBatch(double[][][] data) {
        double[][] out = new double[firstDim * secondDim][thirdDim];
        int row = 0;
        for (int i = 0; i < firstDim; i++) {
            for (int j = 0; j < secondDim; j++) {
                System.arraycopy(data[i][j], 0, out[row ++], 0, thirdDim);
            }
        }
        return out;
    }




    @Override
    public double[][][] backwardBatch(double[][] data) {
        double[][][] out = new double[firstDim][secondDim][thirdDim];
        int row = 0;
        for (int i = 0; i < firstDim; i++) {
            for (int j = 0; j < secondDim; j++) {
                System.arraycopy(data[row], 0, out[i][j], 0, thirdDim);
                row++;
            }
        }
        return out;
    }


}