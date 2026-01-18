package com.oraen.box.otorch.transformer.tokenizer;

import lombok.Getter;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

@Getter
public class ByteEncoder {

    private static final String[] BYTES_TO_UNICODE;
    private static final Map<String, Integer> UNICODE_TO_BYTES;

    public ByteEncoder() {
    }


    static {
        // 初始化 256 个字节到 Unicode 的映射
        BYTES_TO_UNICODE = new String[256];
        UNICODE_TO_BYTES = new HashMap<>();

        // 可打印 ASCII 范围 (33~126)
        int n = 0;
        for (int i = 33; i < 127; i++) {
            BYTES_TO_UNICODE[i] = String.valueOf((char) i);
            UNICODE_TO_BYTES.put(String.valueOf((char) i), i);
            n++;
        }

        // 控制字符和其他字节 (0~32, 127~255)
        for (int i = 0; i < 256; i++) {
            if (BYTES_TO_UNICODE[i] == null) {
                // 使用 Unicode 私有区字符 (从 256 开始)
                BYTES_TO_UNICODE[i] = String.valueOf((char) (0xE000 + n));
                UNICODE_TO_BYTES.put(BYTES_TO_UNICODE[i], i);
                n++;
            }
        }

    }

    public static String encodeSingle(byte b) {
        int i = b & 0xFF;
        return BYTES_TO_UNICODE[i];
    }

    public static String encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        return encode(bytes);
    }

    public static String encode(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(encodeSingle(b));
        }
        return sb.toString();
    }

    public static byte[] decode(String text) {
        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++) {
            String ch = String.valueOf(text.charAt(i));
            bytes[i] = UNICODE_TO_BYTES.get(ch).byteValue();
        }
        return bytes;
    }

    public static void main(String[] args) {
        String original = "Hello, 世界!阿萨德 123";
        byte[] bytes = original.getBytes();
        String encoded = ByteEncoder.encode(bytes);
        byte[] decode = ByteEncoder.decode(encoded);
        String re = new String(decode, java.nio.charset.StandardCharsets.UTF_8);

        System.out.println("Original: " + original);
        System.out.println("Encoded: " + encoded);
        System.out.println("re: " + re);

    }
}
