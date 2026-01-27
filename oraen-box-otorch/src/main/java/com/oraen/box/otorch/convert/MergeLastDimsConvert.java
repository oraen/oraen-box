package com.oraen.box.otorch.convert;

import com.oraen.box.otorch.Layer;


/**
 *  Merge last two dims: [a][b][c] -> [a][b*c]
 */
public class MergeLastDimsConvert implements Layer<double[][], double[]> {
    private final int firstDim;
    private final int secondDim;
    private final int thirdDim;

    public MergeLastDimsConvert(int firstDim, int secondDim, int thirdDim) {
        this.firstDim = firstDim;
        this.secondDim = secondDim;
        this.thirdDim = thirdDim;
    }


    @Override
    public double[][] forwardBatch(double[][][] data) {
        double[][] out = new double[firstDim][secondDim * thirdDim];
        for (int i = 0; i < firstDim; i++) {
            int idx = 0;
            for (int j = 0; j < secondDim; j++) {
                for (int k = 0; k < thirdDim; k++) {
                    out[i][idx++] = data[i][j][k];
                }
            }
        }
        return out;
    }


    @Override
    public double[][][] backwardBatch(double[][] data) {
        double[][][] out = new double[firstDim][secondDim][thirdDim];
        for (int i = 0; i < firstDim; i++) {
            int idx = 0;
            for (int j = 0; j < secondDim; j++) {
                for (int k = 0; k < thirdDim; k++) {
                    out[i][j][k] = data[i][idx++];
                }
            }
        }
        return out;
    }

}