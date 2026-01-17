package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.Tokenizer;

import java.util.*;

/**
 * Base BPE Tokenizer (word-level with word boundary marker)
 */
public class BPETokenizer implements Tokenizer {

    BPEVocabInfo BPEVocabInfo;

    public BPETokenizer(BPEVocabInfo BPEVocabInfo) {
        this.BPEVocabInfo = BPEVocabInfo;
    }

    @Override
    public int[] encode(String text) {
        List<Integer> output = new ArrayList<>();

        // 按一个或多个空白字符切分，自动忽略前导/中间/尾随空白
        String[] words = text.split("\\s+");

        for (String word : words) {
            output.addAll(encodeWord(word));
        }

        return output.stream().mapToInt(Integer::intValue).toArray();
    }

    protected List<Integer> encodeWord(String word) {
        int wordBoundaryId = BPEVocabInfo.getWordBoundaryId();
        int unkId = BPEVocabInfo.getUnkId();
        Map<String, Integer> vocab = BPEVocabInfo.getVocab();

        List<Integer> symbols = new ArrayList<>();
        symbols.add(wordBoundaryId);
        // character-level init
        for (char c : word.toCharArray()) {
            String s = String.valueOf(c);
            symbols.add(vocab.getOrDefault(s, unkId));
        }

        return bpeMerge(symbols);
    }

    // =========================
    // BPE merge (ID-based)
    // =========================
    protected List<Integer> bpeMerge(List<Integer> symbols) {
        Map<IntPair, Integer> bpeRanks = BPEVocabInfo.getBpeRanks();
        Map<String, Integer> vocab = BPEVocabInfo.getVocab();
        String[] idToToken = BPEVocabInfo.getIdToToken();

        while (true) {
            // 查找最优 pair 及其首次出现位置
            IntPair bestPair = null;
            int bestRank = Integer.MAX_VALUE;
            // 记录位置
            int bestPos = -1;
            int end = symbols.size() - 1;

            for (int i = 0; i < end; i++) {
                IntPair pair = new IntPair(symbols.get(i), symbols.get(i + 1));
                Integer rank = bpeRanks.get(pair);
                if (rank == null) continue;

                // 提前检查 merged 是否在 vocab 中（避免无效合并）
                String merged = idToToken[pair.first] + idToToken[pair.second];
                if (!vocab.containsKey(merged)) continue;

                if (rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                    bestPos = i; // 记录位置
                }
            }

            if (bestPair == null) break;

            // 构建新列表：[0, bestPos) + [mergedId] + (bestPos+2, end]
            String merged = idToToken[bestPair.first] + idToToken[bestPair.second];
            Integer mergedId = vocab.get(merged);

            List<Integer> newSymbols = new ArrayList<>(symbols.size() - 1); // 预估大小
            // 添加前半部分
            newSymbols.addAll(symbols.subList(0, bestPos));
            // 添加合并后的 token
            newSymbols.add(mergedId);
            // 添加后半部分（跳过两个元素）
            newSymbols.addAll(symbols.subList(bestPos + 2, symbols.size()));

            symbols = newSymbols;
        }
        return symbols;
    }



    @Override
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        String[] idToToken = BPEVocabInfo.getIdToToken();
        String wordBoundary = BPEVocabInfo.getWordBoundary();

        for (int id : tokenIds) {
            String token = idToToken[id];

            if (token.equals(wordBoundary)) {
                sb.append(' ');
            } else {
                sb.append(token);
            }
        }

        return sb.toString().trim();
    }


}
