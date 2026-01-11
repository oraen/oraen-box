package com.oraen.box.otorch;


public interface Layer<T, U> {

    U[] forwardBatch(T[] data);

    T[] backwardBatch(U[] gradOutputBatch);
}

