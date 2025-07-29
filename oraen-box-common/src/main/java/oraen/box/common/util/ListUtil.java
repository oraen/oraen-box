package oraen.box.common.util;

import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("unchecked")
public class ListUtil {

    public static <T> List<T> of(T... ts) {
        List<T> list = new ArrayList<>(ts.length);
        for (T t : ts) {
            if (t != null) {
                list.add(t);
            }
        }
        return list;
    }
}
