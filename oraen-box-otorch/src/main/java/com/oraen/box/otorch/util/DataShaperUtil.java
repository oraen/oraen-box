package com.oraen.box.otorch.util;

import java.util.ArrayList;
import java.util.List;

public class DataShaperUtil {

    public static List<String> divide(String str){
        List<String> array = new ArrayList<>(str.length());
        for(int i=0;i<str.length();i++){
            array.add(String.valueOf(str.charAt(i)));
        }
        return array;
    }


    public static List<String> mergePair(List<String> source, String first, String second) {
        List<String> result = new ArrayList<>(source.size());

        for (int i = 0; i < source.size(); i++) {
            // 匹配 pair
            if (i < source.size() - 1 && source.get(i).equals(first) && source.get(i + 1).equals(second)) {
                // 合并
                result.add(first + second);
                i++; // 跳过 second（non-overlap）
            } else {
                result.add(source.get(i));
            }
        }

        return result;
    }
}
