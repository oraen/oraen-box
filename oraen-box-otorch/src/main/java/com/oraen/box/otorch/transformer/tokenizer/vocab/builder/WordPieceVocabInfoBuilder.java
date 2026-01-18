package com.oraen.box.otorch.transformer.tokenizer.vocab.builder;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.tokenizer.vocab.WordPieceVocabInfo;

import java.util.*;

/**
 * 完整修正版 WordPieceVocabInfoBuilder
 *
 * 改进点：
 * 1. 高频词加入初始词表，减少 <unk> 出现
 * 2. 拆分逻辑 fallback 到单字符，防止死循环
 * 3. 合并逻辑忽略 <unk> token，只添加新增 token
 * 4. 增加最大迭代次数，安全退出
 */
public class WordPieceVocabInfoBuilder {

    private int vocabSize = 30000;
    private String unk = "<unk>";
    private String continuationPrefix = "##";
    private String corpus;
    private int highFreqThreshold = 5; // 高频词阈值，可调整

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
                for (char c : word.toCharArray()) allChars.add(c);
            }
        }

        WordPieceVocabInfo vocabInfo = new WordPieceVocabInfo();
        vocabInfo.addUnk(unk);
        vocabInfo.setContinuationPrefix(continuationPrefix);

        // 2. 初始化词表：所有字符
        for (char c : allChars) vocabInfo.addToken(String.valueOf(c));

        // 加入语料中高频词作为初始 token
        for (Map.Entry<String,Integer> entry : wordFreqs.entrySet()) {
            String word = entry.getKey();
            int freq = entry.getValue();
            if (freq >= highFreqThreshold && word.length() > 1) {
                vocabInfo.addToken(word);
            }
        }

        // 3. 循环合并子词
        int iterations = 0;
        int maxIterations = 100000; // 安全退出
        while (vocabInfo.getVocabSize() < vocabSize && iterations < maxIterations) {
            iterations++;
            Map<IntPair, Integer> pairFreqs = new HashMap<>();

            // 统计所有词中相邻子词对出现次数
            for (Map.Entry<String,Integer> entry : wordFreqs.entrySet()) {
                String word = entry.getKey();
                int freq = entry.getValue();
                List<String> tokens = tokenizeWithCurrentVocab(word, vocabInfo);

                for (int i = 0; i < tokens.size()-1; i++) {
                    String first = tokens.get(i);
                    String second = tokens.get(i+1);
                    if (first.equals(unk) || second.equals(unk)) continue; // 忽略 <unk>
                    int id1 = vocabInfo.getId(first);
                    int id2 = vocabInfo.getId(second);
                    pairFreqs.merge(new IntPair(id1, id2), freq, Integer::sum);
                }
            }

            if (pairFreqs.isEmpty()) break;

            // 找频率最高的 pair
            IntPair bestPair = Collections.max(pairFreqs.entrySet(), Map.Entry.comparingByValue()).getKey();
            String first = vocabInfo.getToken(bestPair.first);
            String second = vocabInfo.getToken(bestPair.second);

            boolean firstIsContinuation = first.startsWith(continuationPrefix);
            boolean secondIsContinuation = second.startsWith(continuationPrefix);

            String cleanFirst = firstIsContinuation ? first.substring(continuationPrefix.length()) : first;
            String cleanSecond = secondIsContinuation ? second.substring(continuationPrefix.length()) : second;

            String merged = cleanFirst + cleanSecond;
            if (firstIsContinuation) merged = continuationPrefix + merged;

            // 仅在 vocab 中不存在时添加
            if (!vocabInfo.containsToken(merged)) {
                vocabInfo.addToken(merged);
            } else {
                // 无法新增 token，退出循环
                break;
            }
        }

        return vocabInfo;
    }

    /**
     * 使用当前词表将单词拆分成子词（最大匹配）
     * 如果找不到匹配，则 fallback 为单字符
     */
    private List<String> tokenizeWithCurrentVocab(String word, WordPieceVocabInfo vocabInfo) {
        List<String> result = new ArrayList<>();
        int start = 0;
        int len = word.length();
        while (start < len) {
            int end = len;
            String bestMatch = null;
            while (start < end) {
                String sub = word.substring(start, end);
                if (start > 0) sub = continuationPrefix + sub;
                if (vocabInfo.containsToken(sub)) {
                    bestMatch = sub;
                    break;
                }
                end--;
            }

            if (bestMatch == null) {
                // fallback：单字符 token
                String sub = String.valueOf(word.charAt(start));
                if (start > 0) sub = continuationPrefix + sub;
                result.add(sub);
                start++;
            } else {
                result.add(bestMatch);
                start = (bestMatch.startsWith(continuationPrefix) ? bestMatch.length()-continuationPrefix.length() : bestMatch.length()) + start;
            }
        }
        return result;
    }
}
