package com.oraen.box.common.util;

import org.apache.commons.lang3.StringUtils;

import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

public class EnumUtil {

    public static <T extends Enum<T>, V> T getEnum(Class<T> enumClass, Function<T, V> match, V target, T defaultValue) {
        for(T e : enumClass.getEnumConstants()){
            if(Objects.equals(match.apply(e), target)) {
                return e;
            }
        }
        return defaultValue;
    }

    public static <T extends Enum<T>, V> T getEnum(Class<T> enumClass, Function<T, V> match, V target) {
        return getEnum(enumClass, match, target, null);
    }

    public static <T extends Enum<T>, V> Optional<T> getEnumOptional(Class<T> enumClass, Function<T, V> match, V target) {
        return Optional.ofNullable(getEnum(enumClass, match, target));
    }

    public static <T extends Enum<T>> T getEnumIgnoreCase(Class<T> enumClass, Function<T, String> match, String target, T defaultValue) {
        for(T e : enumClass.getEnumConstants()){
            if(StringUtils.equalsIgnoreCase(match.apply(e), target)) {
                return e;
            }
        }
        return defaultValue;
    }

    public static <T extends Enum<T>> T getEnumIgnoreCase(Class<T> enumClass, Function<T, String> match, String target) {
        return getEnumIgnoreCase(enumClass, match, target, null);
    }

    public static <T extends Enum<T>> Optional<T> getEnumOptionalIgnoreCase(Class<T> enumClass, Function<T, String> match, String target) {
        return Optional.ofNullable(getEnumIgnoreCase(enumClass, match, target));
    }


}
