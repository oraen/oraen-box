package com.oraen.box.otorch.transformer.tokenizer;

import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

@Getter
public class ByteEncoder {

    private static final String[] BYTES_TO_UNICODE;
    private static final Map<String, Integer> UNICODE_TO_BYTES;
    private static final int SPACE_BYTE = 32;

    private final String byteSpace;

    public ByteEncoder() {
        this("Ġ");
    }

    public ByteEncoder(String byteSpace) {
        this.byteSpace = byteSpace;
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
                BYTES_TO_UNICODE[i] = String.valueOf((char) (256 + n));
                UNICODE_TO_BYTES.put(BYTES_TO_UNICODE[i], i);
                n++;
            }
        }

    }

    public String encode(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            int i = b & 0xFF;
            sb.append(i == SPACE_BYTE ? byteSpace : BYTES_TO_UNICODE[i]);
        }
        return sb.toString();
    }

    public byte[] decode(String text) {
        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++) {
            String ch = String.valueOf(text.charAt(i));
            if (ch.equals(byteSpace)) {
                bytes[i] = SPACE_BYTE;
            }else{
                bytes[i] = UNICODE_TO_BYTES.get(ch).byteValue();
            }
        }
        return bytes;
    }
}
