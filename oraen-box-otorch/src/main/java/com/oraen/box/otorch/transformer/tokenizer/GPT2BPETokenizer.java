package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;

import java.util.Map;

public class GPT2BPETokenizer extends BPETokenizer {

    public GPT2BPETokenizer(
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
                "Ġ"   // GPT-2 uses Ġ as word boundary
        );
    }

    @Override
    public int[] encode(String text) {
        /*
         * GPT-2 的“空格即词首”语义：
         *  hello world
         *  -> hello Ġworld
         */
        if (text.startsWith(" ")) {
            text = "Ġ" + text.substring(1);
        }
        text = text.replace(" ", "Ġ");

        return super.encode(text);
    }

    @Override
    public String decode(int[] tokenIds) {
        String text = super.decode(tokenIds);
        return text.replace("Ġ", " ");
    }
}
