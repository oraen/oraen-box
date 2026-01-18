package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.otorch.transformer.tokenizer.vocab.BPEVocabInfo;

public class GPT2BPETokenizer extends ByteLevelBPETokenizer {
    public GPT2BPETokenizer(BPEVocabInfo bpeVocabInfo) {
        super(bpeVocabInfo);
    }


    @Override
    public int[] encode(String text) {

        String wordBoundary = bpeVocabInfo.getWordBoundary();

        // GPT-2 规则 1：文本开头强制一个空格
        if (!text.startsWith(" ")) {
            text = " " + text;
        }


        // 注意：这里只处理语义，不做 byte 编码
        String gpt2Text = text.replace(" ", wordBoundary);

        // 后续流程完全复用 ByteLevelBPETokenizer
        return super.encode(gpt2Text);
    }

    @Override
    public String decode(int[] tokenIds) {
        // GPT-2 的 decode 行为与 ByteLevel 是一致的
        return super.decode(tokenIds);
    }

}
