package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;

import java.nio.charset.StandardCharsets;
import java.util.Map;

public class ByteLevelBPETokenizer extends BPETokenizer {

    private final ByteEncoder byteEncoder = new ByteEncoder();

    public ByteLevelBPETokenizer(
            Map<IntPair, Integer> bpeRanks,
            Map<String, Integer> vocab
    ) {
        super(
                bpeRanks,
                vocab,
                vocab.get("<pad>"),
                vocab.get("<bos>"),
                vocab.get("<eos>"),
                vocab.get("<unk>"),
                "<unk>",
                "Ġ"   // GPT-2 / ByteLevel BPE 的 word boundary
        );
    }

    @Override
    public int[] encode(String text) {
        // 1. UTF-8 bytes
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);

        // 2. byte -> printable chars
        String encoded = byteEncoder.encode(bytes);

        // 3. normal BPE encode
        return super.encode(encoded);
    }

    @Override
    public String decode(int[] tokenIds) {
        // 1. normal BPE decode (得到 byte-level string)
        String encoded = super.decode(tokenIds);

        // 2. printable chars -> bytes
        byte[] bytes = byteEncoder.decode(encoded);

        // 3. bytes -> UTF-8 string
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
