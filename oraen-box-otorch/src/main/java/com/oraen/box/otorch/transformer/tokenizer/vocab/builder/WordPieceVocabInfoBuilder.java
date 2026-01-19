package com.oraen.box.otorch.transformer.tokenizer.vocab.builder;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.tokenizer.vocab.WordPieceVocabInfo;

import java.util.*;

/**
 * 修正版 WordPieceVocabInfoBuilder
 *
 * 修复内容：
 * 1. 修正 tokenizeWithCurrentVocab 中的 start 更新逻辑
 * 2. 优化贪心匹配效率
 * 3. 改进高频词初始化策略
 * 4. 修复合并 token 已存在时的处理逻辑
 */
public class WordPieceVocabInfoBuilder {

    private int vocabSize = 30000;
    private String unk = "<unk>";
    private String continuationPrefix = "##";
    private String corpus;
    private int highFreqThreshold = 10; // 提高阈值，只加入真正高频的词

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

        // 2. 初始化词表：添加所有单字符
        for (char c : allChars) {
            vocabInfo.addToken(String.valueOf(c));
        }

        // 可选：加入极高频的短词（长度2-3）作为优化
        // 这可以减少训练迭代，但要谨慎使用
        for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
            String word = entry.getKey();
            int freq = entry.getValue();
            // 只加入长度2-3且频率极高的词
            if (freq >= highFreqThreshold && word.length() >= 2 && word.length() <= 3) {
                if (!vocabInfo.containsToken(word)) {
                    vocabInfo.addToken(word);
                }
            }
        }

        // 3. 迭代合并子词对
        int iterations = 0;
        int maxIterations = 100000;
        int noProgressCount = 0; // 连续无进展次数

        while (vocabInfo.getVocabSize() < vocabSize && iterations < maxIterations) {
            iterations++;
            Map<IntPair, Integer> pairFreqs = new HashMap<>();

            // 统计所有词中相邻子词对的出现次数
            for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
                String word = entry.getKey();
                int freq = entry.getValue();
                List<String> tokens = tokenizeWithCurrentVocab(word, vocabInfo);

                // 统计相邻 token 对
                for (int i = 0; i < tokens.size() - 1; i++) {
                    String first = tokens.get(i);
                    String second = tokens.get(i + 1);

                    // 忽略包含 <unk> 的 pair
                    if (first.equals(unk) || second.equals(unk)) {
                        continue;
                    }

                    int id1 = vocabInfo.getId(first);
                    int id2 = vocabInfo.getId(second);
                    pairFreqs.merge(new IntPair(id1, id2), freq, Integer::sum);
                }
            }

            // 没有可合并的 pair，退出
            if (pairFreqs.isEmpty()) {
                break;
            }

            // 找出频率最高的 pair
            IntPair bestPair = Collections.max(pairFreqs.entrySet(),
                    Map.Entry.comparingByValue()).getKey();

            String first = vocabInfo.getToken(bestPair.first);
            String second = vocabInfo.getToken(bestPair.second);

            // 构造合并后的 token
            String merged = mergeTokens(first, second);

            // 如果合并后的 token 已存在，跳过这次合并，继续尝试其他 pair
            if (vocabInfo.containsToken(merged)) {
                noProgressCount++;
                // 如果连续多次无法添加新 token，可能陷入循环，退出
                if (noProgressCount > 100) {
                    break;
                }
                continue;
            }

            // 添加新 token
            vocabInfo.addToken(merged);
            noProgressCount = 0; // 重置无进展计数
        }

        return vocabInfo;
    }

    /**
     * 合并两个 token，处理 continuation prefix
     */
    private String mergeTokens(String first, String second) {
        boolean firstIsContinuation = first.startsWith(continuationPrefix);
        boolean secondIsContinuation = second.startsWith(continuationPrefix);

        // 去掉前缀
        String cleanFirst = firstIsContinuation ?
                first.substring(continuationPrefix.length()) : first;
        String cleanSecond = secondIsContinuation ?
                second.substring(continuationPrefix.length()) : second;

        // 合并
        String merged = cleanFirst + cleanSecond;

        // 如果第一个 token 有 continuation prefix，合并结果也要有
        if (firstIsContinuation) {
            merged = continuationPrefix + merged;
        }

        return merged;
    }

    /**
     * 使用当前词表对单词进行分词（贪心最长匹配）
     *
     * @param word 待分词的单词
     * @param vocabInfo 当前词表
     * @return 分词结果
     */
    private List<String> tokenizeWithCurrentVocab(String word, WordPieceVocabInfo vocabInfo) {
        List<String> result = new ArrayList<>();
        int start = 0;
        int len = word.length();

        while (start < len) {
            int end = len;
            String bestMatch = null;

            // 从最长子串开始尝试，贪心匹配
            while (start < end) {
                String sub = word.substring(start, end);

                // 非首个子词需要加 continuation prefix
                if (start > 0) {
                    sub = continuationPrefix + sub;
                }

                if (vocabInfo.containsToken(sub)) {
                    bestMatch = sub;
                    break; // 找到最长匹配，停止
                }

                end--; // 缩短子串继续尝试
            }

            if (bestMatch == null) {
                // 没找到匹配，fallback 到单字符
                String fallback = String.valueOf(word.charAt(start));
                if (start > 0) {
                    fallback = continuationPrefix + fallback;
                }

                // 如果连单字符都不在词表中，使用 <unk>
                if (vocabInfo.containsToken(fallback)) {
                    result.add(fallback);
                } else {
                    result.add(unk);
                }
                start++;
            } else {
                result.add(bestMatch);

                // 关键修复：正确计算前进的字符数
                // 如果 bestMatch 有 prefix，实际字符数要减去 prefix 长度
                int actualChars = bestMatch.startsWith(continuationPrefix) ?
                        bestMatch.length() - continuationPrefix.length() :
                        bestMatch.length();

                start += actualChars;
            }
        }

        return result;
    }
}