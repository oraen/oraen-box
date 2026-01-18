package com.oraen.box.otorch.transformer.tokenizer.vocab.builder;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.tokenizer.vocab.BPEVocabInfo;
import com.oraen.box.otorch.transformer.tokenizer.ByteEncoder;
import com.oraen.box.otorch.util.DataShaperUtil;
import com.oraen.box.otorch.transformer.Tokenizer;

import java.util.*;

/**
 * BPEByteLevelVocabInfoBuilder
 *
 * 作用：
 *  - 从原始 corpus（String）
 *  - 构建 Byte-level BPE 所需的：
 *      1. vocab（token -> id）
 *      2. bpeRanks（pair(idA, idB) -> 合并优先级）
 *
 */
public class BPEByteLevelVocabInfoBuilder implements BPEVocabInfoBuilder{

    /** 目标词表大小 */
    private int vocabSize = 30000;

    /** 未登录词 */
    private String unk = "<unk>";

    private String corpus;

    public static BPEByteLevelVocabInfoBuilder builder() {
        return new BPEByteLevelVocabInfoBuilder();
    }

    public BPEByteLevelVocabInfoBuilder vocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }

    public BPEByteLevelVocabInfoBuilder unk(String unk) {
        this.unk = unk;
        return this;
    }

    public BPEByteLevelVocabInfoBuilder corpus(String corpus) {
        this.corpus = corpus;
        return this;
    }


    /**
     * 构建 Byte-level BPE vocab
     */
    public BPEVocabInfo build() {

        // key: token 序列（byte token list）
        // value: 出现次数（这里只有 1，但保留通用结构）
        Map<List<String>, Integer> wordFreqs = new HashMap<>();

        List<String> tokens = DataShaperUtil.divide(ByteEncoder.encode(corpus));

        wordFreqs.put(tokens, 1);

        BPEVocabInfo bpeVocabInfo = new BPEVocabInfo(Tokenizer.InputMode.BYTE);

        // special token
        bpeVocabInfo.addUnk(unk);

        // 初始化 256 个 byte token
        for (int i = 0; i < 256; i++) {
            bpeVocabInfo.addToken(ByteEncoder.encodeSingle((byte) i));
        }

        // BPE 主循环
        while (bpeVocabInfo.getVocabSize() < vocabSize) {

            Map<IntPair, Integer> pairFreq = new HashMap<>();

            // 统计 pair 频次
            for (Map.Entry<List<String>, Integer> entry : wordFreqs.entrySet()) {
                List<String> seq = entry.getKey();
                int freq = entry.getValue();

                for (int i = 0; i < seq.size() - 1; i++) {
                    int a = bpeVocabInfo.getId(seq.get(i));
                    int b = bpeVocabInfo.getId(seq.get(i + 1));
                    pairFreq.merge(new IntPair(a, b), freq, Integer::sum);
                }
            }

            if (pairFreq.isEmpty()) break;

            // 选最高频 pair
            IntPair best = Collections.max(pairFreq.entrySet(), Map.Entry.comparingByValue()).getKey();
            String merged = bpeVocabInfo.getToken(best.first) + bpeVocabInfo.getToken(best.second);
            bpeVocabInfo.addToken(merged);
            bpeVocabInfo.nextRank(best);

            // 替换语料
            Map<List<String>, Integer> newWordFreqs = new HashMap<>();
            for (Map.Entry<List<String>, Integer> entry : wordFreqs.entrySet()) {
                List<String> token = entry.getKey();
                Integer num = entry.getValue();
                List<String> newTokens = DataShaperUtil.mergePair(token, bpeVocabInfo.getToken(best.first), bpeVocabInfo.getToken(best.second));
                newWordFreqs.merge(newTokens, num, Integer::sum);
            }

            wordFreqs = newWordFreqs;
        }

        return bpeVocabInfo;
    }


}
