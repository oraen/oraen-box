package com.oraen.box.common.function;

public interface TripleFunction<T, Y, U, I> {

    I apply(T t, Y y, U u);

}
