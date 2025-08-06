package oraen.box.common.util;

import java.util.Collection;

public class CollectionUtil {

    public static  boolean isEmpty(Collection<?> collection) {
        return collection == null || collection.isEmpty();
    }

    public static int size(Collection<?> collection) {
        return collection == null ? 0 : collection.size();
    }
}
