package com.oraen.box.otorch.transformer;

import com.oraen.box.otorch.*;
import com.oraen.box.otorch.activation.ActivationFunctionIndependently;
import com.oraen.box.otorch.activation.ActivationLayerAdapter;
import com.oraen.box.otorch.component.CommonAffineLayer;
import com.oraen.box.otorch.initializer.XavierNormalInitializer;
import com.oraen.box.otorch.optimizer.AdamWOptimizer;
import lombok.Data;

@Data
public class FeedForwardLayer implements Layer<double[], double[]>, Learnable {

    private AffineLayer fc1;
    private ActivationLayerAdapter act;
    private AffineLayer fc2;

    public FeedForwardLayer(AffineLayer fc1, ActivationFunctionIndependently activationFunction, AffineLayer fc2){
        this.fc1 = fc1;
        this.act = new ActivationLayerAdapter(activationFunction);
        this.fc2 = fc2;
    }

    public FeedForwardLayer(int modelDim, int hiddenDim, ActivationFunctionIndependently activationFunction) {
        this.fc1 = new CommonAffineLayer(modelDim, hiddenDim, XavierNormalInitializer.INSTANCE, new AdamWOptimizer(hiddenDim, modelDim));
        this.act = new ActivationLayerAdapter(activationFunction);
        this.fc2 = new CommonAffineLayer(hiddenDim, modelDim, XavierNormalInitializer.INSTANCE, new AdamWOptimizer(modelDim, hiddenDim));
    }


    @Override
    public double[][] forwardBatch(double[][] data) {
        double[][] fc1Out = fc1.forwardBatch(data);
        double[][] actOut = act.forwardBatch(fc1Out);
        return fc2.forwardBatch(actOut);
    }

    @Override
    public double[][] backwardBatch(double[][] gradOutputBatch) {
        double[][] fc2GradInput = fc2.backwardBatch(gradOutputBatch);
        double[][] actGradInput = act.backwardBatch(fc2GradInput);
        return fc1.backwardBatch(actGradInput);
    }

    @Override
    public void updateParameters() {
        fc1.updateParameters();
        fc2.updateParameters();
    }
}