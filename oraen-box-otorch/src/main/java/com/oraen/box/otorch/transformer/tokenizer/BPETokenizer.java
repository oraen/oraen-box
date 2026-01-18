package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.Tokenizer;
import com.oraen.box.otorch.transformer.tokenizer.vocab.BPEVocabInfo;

import java.util.*;

/**
 * Base BPE Tokenizer (word-level with word boundary marker)
 */
public abstract class BPETokenizer implements Tokenizer {

    BPEVocabInfo bpeVocabInfo;

    public BPETokenizer(BPEVocabInfo bpeVocabInfo) {
        if(bpeVocabInfo.getInputMode() != this.getInputMode()){
            throw new IllegalArgumentException("BPEVocabInfo input mode mismatch. Expected: " + this.getInputMode() + ", but got: " + bpeVocabInfo.getInputMode());
        }
        this.bpeVocabInfo = bpeVocabInfo;
    }


    protected List<Integer> bpeMerge(List<Integer> symbols) {

        while (true) {
            // 查找最优 pair 及其首次出现位置
            IntPair bestPair = null;
            int bestRank = Integer.MAX_VALUE;
            // 记录位置
            int bestPos = -1;
            int end = symbols.size() - 1;

            for (int i = 0; i < end; i++) {
                IntPair pair = new IntPair(symbols.get(i), symbols.get(i + 1));
                int rank = bpeVocabInfo.getRank(pair);
                if (rank == Integer.MAX_VALUE) continue;

                // 提前检查 merged 是否在 vocab 中（避免无效合并）
                String merged = bpeVocabInfo.getToken(pair.first) + bpeVocabInfo.getToken(pair.second);
                if (!bpeVocabInfo.containsToken(merged)) continue;

                if (rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                    bestPos = i; // 记录位置
                }
            }

            if (bestPair == null) break;

            // 构建新列表：[0, bestPos) + [mergedId] + (bestPos+2, end]
            String merged = bpeVocabInfo.getToken(bestPair.first) + bpeVocabInfo.getToken(bestPair.second);
            Integer mergedId = bpeVocabInfo.getId(merged);

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


}
