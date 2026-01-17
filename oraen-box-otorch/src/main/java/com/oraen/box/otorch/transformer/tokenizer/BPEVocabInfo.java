package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;
import lombok.Getter;

import java.util.Collections;
import java.util.Map;

@Getter
public class BPEVocabInfo {
    Map<IntPair, Integer> bpeRanks;
    Map<String, Integer> vocab;
    int unkId;
    int wordBoundaryId;
    String unk;
    String wordBoundary;
    String[] idToToken;

    private BPEVocabInfo(Map<IntPair, Integer> bpeRanks, Map<String, Integer> vocab) {
        this.bpeRanks = bpeRanks;
        this.vocab = vocab;
        int maxId = Collections.max(vocab.values());
        this.idToToken = new String[maxId + 1];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            idToToken[e.getValue()] = e.getKey();
        }
    }

    public BPEVocabInfo(Map<IntPair, Integer> bpeRanks, Map<String, Integer> vocab, String unk, String wordBoundary) {
        this(bpeRanks, vocab);
        this.unk = unk;
        this.wordBoundary = wordBoundary;
        this.unkId = vocab.get(unk);
        this.wordBoundaryId = vocab.get(wordBoundary);
    }

    public BPEVocabInfo(Map<IntPair, Integer> bpeRanks, Map<String, Integer> vocab, int unkId, int wordBoundaryId) {
        this(bpeRanks, vocab);
        this.unkId = unkId;
        this.wordBoundaryId = wordBoundaryId;
        this.unk = idToToken[unkId];
        this.wordBoundary = idToToken[wordBoundaryId];
    }
}
