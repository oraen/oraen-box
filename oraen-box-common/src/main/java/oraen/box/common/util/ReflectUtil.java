package oraen.box.common.util;

public class ReflectUtil {

    public static String getSimClassName(Object obj) {
        return getSimClassName(obj.getClass());
    }

    public static String getSimClassName(Class<?> clazz) {
        if (clazz == null) {
            throw new IllegalArgumentException("Class cannot be null");
        }
        return clazz.getSimpleName();
    }

}
