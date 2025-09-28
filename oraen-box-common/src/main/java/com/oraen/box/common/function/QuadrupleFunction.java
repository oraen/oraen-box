package com.oraen.box.common.function;

public interface QuadrupleFunction<T, Y, U, I, O> {

    O apply(T t, Y y, U u, I i);

}
