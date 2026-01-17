package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;

import java.nio.charset.StandardCharsets;
import java.util.Map;

public class ByteLevelBPETokenizer extends BPETokenizer {

    public static final String WORD_BOUNDARY = "Ġ";
    public static final String UNK = "<unk>";

    private final ByteEncoder byteEncoder = new ByteEncoder();

    public ByteLevelBPETokenizer(Map<IntPair, Integer> bpeRanks, Map<String, Integer> vocab) {
        super(new BPEVocabInfo(bpeRanks, vocab, UNK, WORD_BOUNDARY));
    }

    @Override
    public int[] encode(String text) {
        /**
         * 本质就是把空格替换为"Ġ"，并且把学习字符的行为转变为学习对应字符编码（主要针对中文，标签等其他字符）
         */
        // 1. UTF-8 bytes
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);

        // 2. byte -> printable chars
        String encoded = byteEncoder.encode(bytes);

        // 3. normal BPE encode
        return super.encode(encoded);
    }

    @Override
    public String decode(int[] tokenIds) {
        String[] idToToken = BPEVocabInfo.getIdToToken();

        // 1. 直接拼接 token 字符串（不替换 Ġ！）
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            sb.append(idToToken[id]); // 原样拼接
        }
        String encoded = sb.toString();

        // 2. byte decode
        byte[] bytes = byteEncoder.decode(encoded);

        // 3. UTF-8 restore
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
