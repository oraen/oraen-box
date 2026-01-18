package com.oraen.box.otorch.transformer.tokenizer.vocab.builder;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.tokenizer.vocab.BPEVocabInfo;
import com.oraen.box.otorch.util.DataShaperUtil;
import com.oraen.box.otorch.transformer.Tokenizer;

import java.util.*;

/**
 * BPEVocabInfoBuilder
 *
 * 作用：
 *  - 从原始语料库（List<String> corpus）中
 *  - 训练出一套 BPE 所需的：
 *      1. vocab（token -> id）
 *      2. bpeRanks（pair(idA, idB) -> 合并优先级）
 *
 * 该 Builder 只负责“训练 / 构建词表信息”
 * 不负责 tokenize / encode / decode
 */
public class BPEWordLevelVocabInfoBuilder implements BPEVocabInfoBuilder{

    /** 目标词表大小（包含 special token + merge 后 token） */
    private int vocabSize = 30000;

    /** 未登录词（unknown token）的字符串形式 */
    private String unk = "<unk>";

    /** 单词边界标记（GPT-2 / Byte-level BPE 常用 Ġ） */
    private String wordBoundary = "Ġ";

    private String corpus;

    public static BPEWordLevelVocabInfoBuilder builder() {
        return new BPEWordLevelVocabInfoBuilder();
    }

    public BPEWordLevelVocabInfoBuilder vocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }

    public BPEWordLevelVocabInfoBuilder unk(String unk) {
        this.unk = unk;
        return this;
    }

    public BPEWordLevelVocabInfoBuilder wordBoundary(String wordBoundary) {
        this.wordBoundary = wordBoundary;
        return this;
    }

    public BPEWordLevelVocabInfoBuilder corpus(String corpus) {
        this.corpus = corpus;
        return this;
    }


    /**
     * 从语料库训练 BPE 词表
     */
    public BPEVocabInfo build() {


        //key 一个词的token序列（例如 ["Ġ", "h", "e", "l", "l", "o"]）， value : 该 token 序列在语料中出现的次数
        Map<List<String>, Integer> wordFreqs = new HashMap<>();

        // 按空白切分成“词”
        for (String word : corpus.split("\\s+")) {

            // 每个词初始化为：
            // [wordBoundary, char1, char2, ...]
            List<String> tokens = new ArrayList<>();
            tokens.add(wordBoundary);
            tokens.addAll(DataShaperUtil.divide(word));

            // 统计该 token 序列的出现频次
            wordFreqs.merge(tokens, 1, Integer::sum);
        }

        BPEVocabInfo bpeVocabInfo = new BPEVocabInfo(Tokenizer.InputMode.WORD);

        // special token 优先占用 id
        bpeVocabInfo.addUnk(unk);
        bpeVocabInfo.addWordBoundary(wordBoundary);

        // 将所有“初始字符 token”加入 vocab
        for (List<String> tokens : wordFreqs.keySet()) {
            for (String t : tokens) {
                //bpeVocabInfo.addToken中会过滤重复token
                bpeVocabInfo.addToken(t);
            }
        }

        while (bpeVocabInfo.getVocabSize() < vocabSize) {

            Map<IntPair, Integer> pairFreq = new HashMap<>();

            // 统计所有 token 序列中的相邻 pair 频次
            for (Map.Entry<List<String>, Integer> wordFreq : wordFreqs.entrySet()) {
                List<String> tokens = wordFreq.getKey();
                int num = wordFreq.getValue();

                int end = tokens.size() - 1;
                for (int i = 0; i < end; i++) {
                    int before = bpeVocabInfo.getId(tokens.get(i));
                    int second = bpeVocabInfo.getId(tokens.get(i + 1));
                    /*
                     * (a, b)：某个相邻 token pair（用 token id 表示）
                     * freq：该 token 序列（word）在语料中出现的次数
                     * 所有词里相同的 pair，把它们的出现次数加起来,所以一次增加freq
                     */
                    pairFreq.merge(new IntPair(before, second), num, Integer::sum);
                }
            }

            // 没有可合并 pair，提前结束
            if (pairFreq.isEmpty()) break;

            // 找出现频率最高的 pair（BPE 的核心）
            IntPair best = Collections.max(pairFreq.entrySet(), Map.Entry.comparingByValue()).getKey();

            // 将 pair 对应的两个token 拼成一个新 token
            String merged = bpeVocabInfo.getToken(best.first) + bpeVocabInfo.getToken(best.second);

            // 新 token 加入 vocab
            bpeVocabInfo.addToken(merged);

            // 记录该 pair 的合并顺序（rank）
            bpeVocabInfo.nextRank(best);

            //用merged token 替换原语料
            Map<List<String>, Integer> newWordFreqs = new HashMap<>();

            for (Map.Entry<List<String>, Integer> wordFreq : wordFreqs.entrySet()) {
                List<String> tokens = wordFreq.getKey();
                Integer num = wordFreq.getValue();
                List<String> newTokens = DataShaperUtil.mergePair(tokens, bpeVocabInfo.getToken(best.first), bpeVocabInfo.getToken(best.second));
                newWordFreqs.merge(newTokens, num, Integer::sum);
            }

            // 用新的 token 序列替换旧的
            wordFreqs = newWordFreqs;
        }

        return bpeVocabInfo;
    }


}
