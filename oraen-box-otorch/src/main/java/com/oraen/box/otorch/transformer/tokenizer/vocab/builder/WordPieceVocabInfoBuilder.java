package com.oraen.box.otorch.transformer.tokenizer.vocab.builder;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.tokenizer.vocab.WordPieceVocabInfo;

import java.util.*;

/**
 * 修正版 WordPieceVocabInfoBuilder
 *
 * 核心修正：
 * 1. 词表构建阶段的 tokenize 不使用 continuation prefix（##）
 * 2. mergeTokens 直接拼接，不处理前缀
 * 3. 初始化仅添加单字符（不加 ## 前缀）
 * 4. 分词输出时的 ## 前缀应在推理阶段处理，不在词表构建中
 */
public class WordPieceVocabInfoBuilder {

    private int vocabSize = 30000;
    private String unk = "<unk>";
    private String continuationPrefix = "##"; // 仅用于推理阶段输出，构建阶段忽略
    private String corpus;
    private int highFreqThreshold = -1;

    public static WordPieceVocabInfoBuilder builder() {
        return new WordPieceVocabInfoBuilder();
    }

    public WordPieceVocabInfoBuilder vocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }

    public WordPieceVocabInfoBuilder unk(String unk) {
        this.unk = unk;
        return this;
    }

    public WordPieceVocabInfoBuilder continuationPrefix(String prefix) {
        this.continuationPrefix = prefix;
        return this;
    }

    public WordPieceVocabInfoBuilder highFreqThreshold(int threshold) {
        this.highFreqThreshold = threshold;
        return this;
    }

    public WordPieceVocabInfoBuilder corpus(String corpus) {
        this.corpus = corpus;
        return this;
    }

    public WordPieceVocabInfo build() {
        // 1. 统计词频
        Map<String, Integer> wordFreqs = new HashMap<>();
        Set<Character> allChars = new HashSet<>();
        for (String word : corpus.split("\\s+")) {
            if (!word.isEmpty()) {
                wordFreqs.merge(word, 1, Integer::sum);
                for (char c : word.toCharArray()) {
                    allChars.add(c);
                }
            }
        }

        WordPieceVocabInfo vocabInfo = new WordPieceVocabInfo();
        vocabInfo.addUnk(unk);
        vocabInfo.setContinuationPrefix(continuationPrefix);

        // 2. 初始化词表：添加所有单字符（原始形式）
        for (char c : allChars) {
            vocabInfo.addTokenIfAbsent(String.valueOf(c));
        }

        // 可选：加入极高频的短词（长度2-3）作为优化
        if(highFreqThreshold > 0){
            for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
                String word = entry.getKey();
                int freq = entry.getValue();
                if (freq >= highFreqThreshold && word.length() >= 2 && word.length() <= 3) {
                    vocabInfo.addTokenIfAbsent(word);
                }
            }
        }

        // 3. 迭代合并子词对
        int iterations = 0;
        int maxIterations = 100000;
        int noProgressCount = 0;

        while (vocabInfo.getVocabSize() < vocabSize && iterations < maxIterations) {
            iterations++;
            Map<IntPair, Integer> pairFreqs = new HashMap<>();

            // 统计所有词中相邻子词对的出现次数
            for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
                String word = entry.getKey();
                int freq = entry.getValue();
                List<String> tokens = tokenizeForBuild(word, vocabInfo); // ← 关键：使用无 ## 的 tokenize

                // 统计相邻 token 对
                for (int i = 0; i < tokens.size() - 1; i++) {
                    String first = tokens.get(i);
                    String second = tokens.get(i + 1);

                    // 忽略包含 <unk> 的 pair， 正常不会发生
                    if (first.equals(unk) || second.equals(unk)) {
                        continue;
                    }

                    int id1 = vocabInfo.getId(first);
                    int id2 = vocabInfo.getId(second);
                    pairFreqs.merge(new IntPair(id1, id2), freq, Integer::sum);
                }
            }

            if (pairFreqs.isEmpty()) {
                break;
            }

            // 找出频率最高的 pair
            IntPair bestPair = Collections.max(pairFreqs.entrySet(), Map.Entry.comparingByValue()).getKey();

            String first = vocabInfo.getToken(bestPair.first);
            String second = vocabInfo.getToken(bestPair.second);
            String merged = mergeTokens(first, second); // ← 简单拼接

            // 如果已存在，跳过
            if (vocabInfo.containsToken(merged)) {
                noProgressCount++;
                if (noProgressCount > 100) {
                    break;
                }
                continue;
            }

            vocabInfo.addTokenIfAbsent(merged);
            noProgressCount = 0;
        }

        return vocabInfo;
    }

    /**
     * 合并两个 token：直接拼接（不处理 continuation prefix）
     */
    private String mergeTokens(String first, String second) {
        return first + second; // 核心修正：不再处理 ##
    }

    /**
     * 专用于词表构建阶段的 tokenize（不使用 continuation prefix）
     * 仅匹配词表中已有的原始 token（如 "play", "ing"）
     */
    private List<String> tokenizeForBuild(String word, WordPieceVocabInfo vocabInfo) {
        List<String> result = new ArrayList<>();
        int start = 0;
        int len = word.length();

        while (start < len) {
            String bestMatch = null;
            int bestEnd = start + 1;

            // 贪心：从最长子串开始尝试，初始化时已加入所有单字符，正常情况下end = start + 1时必有
            for (int end = len; end > start; end--) {
                String sub = word.substring(start, end);
                if (vocabInfo.containsToken(sub)) {
                    bestMatch = sub;
                    bestEnd = end;
                    break;
                }
            }

            // 理论上不应发生，因为单字符已初始化
            if (bestMatch == null) {
                throw new IllegalStateException("Tokenization failed at position " + start + " in word: " + word + ". This usually means single characters were not properly initialized.");
            }

            result.add(bestMatch);
            start = bestEnd;
        }

        return result;
    }
}
