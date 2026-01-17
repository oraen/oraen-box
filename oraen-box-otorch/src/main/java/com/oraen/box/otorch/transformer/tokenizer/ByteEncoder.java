package com.oraen.box.otorch.transformer.tokenizer;

import java.util.HashMap;
import java.util.Map;

public class ByteEncoder {

    private final Map<Integer, String> byteToChar = new HashMap<>();
    private final Map<String, Integer> charToByte = new HashMap<>();

    public ByteEncoder() {
        for (int i = 0; i < 256; i++) {
            String ch = String.valueOf((char) i);
            byteToChar.put(i, ch);
            charToByte.put(ch, i);
        }
    }

    public String encode(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(byteToChar.get(b & 0xff));
        }
        return sb.toString();
    }

    public byte[] decode(String text) {
        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++) {
            bytes[i] = charToByte.get(String.valueOf(text.charAt(i))).byteValue();
        }
        return bytes;
    }
}