package com.oraen.box.otorch;


public interface Layer<T, U> {

    U[] forwardBatch(T[] data);

    U[] backwardBatch(T[] gradOutputBatch);
}

