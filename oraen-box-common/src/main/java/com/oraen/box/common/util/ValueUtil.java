package com.oraen.box.common.util;

import java.util.Objects;

public class ValueUtil {

    @SafeVarargs
    public static<T> T values(T... values) {
        for(T t : values) {
            if(t != null) {
                return t;
            }
        }

        return null;
    }

    @SafeVarargs
    public static <T> boolean valueIn(T t, T... values) {
        for(T value : values) {
            if(Objects.equals(t, value)) {
                return true;
            }
        }
        return false;
    }

    @SafeVarargs
    public static <T> boolean valueNotIn(T t, T... values) {
        return !valueIn(t, values);
    }
}
