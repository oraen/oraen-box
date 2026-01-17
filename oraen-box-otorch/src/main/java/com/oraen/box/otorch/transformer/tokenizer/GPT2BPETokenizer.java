package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;

import java.util.Map;

public class GPT2BPETokenizer extends BPETokenizer {
    public static final String WORD_BOUNDARY = "Ġ";
    public static final String UNK = "<unk>";

    public GPT2BPETokenizer(Map<IntPair, Integer> bpeRanks, Map<String, Integer> vocab) {
        super(new BPEVocabInfo(bpeRanks, vocab, UNK, WORD_BOUNDARY));
    }

    /**
     * GPT2BPE“把整个文本当作一个连续字符串进行 BPE 分词”，其中空格被替换为特殊符号 Ġ，从而保留词边界信息。
     * 正因为如此，xxxĠxxx 这样的跨词组合（如 "theĠend"）会被当作整体学习和合并，这是 GPT-2 能有效建模词间关系的关键。
     */
    @Override
    public int[] encode(String text) {
        if (text.startsWith(" ")) {
            text = WORD_BOUNDARY + text.substring(1);
        }
        text = text.replace(" ", WORD_BOUNDARY);

        return super.encode(text);
    }

}
