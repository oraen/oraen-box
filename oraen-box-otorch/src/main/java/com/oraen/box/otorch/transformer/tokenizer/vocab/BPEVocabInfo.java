package com.oraen.box.otorch.transformer.tokenizer.vocab;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.Tokenizer;
import lombok.Getter;
import lombok.Setter;

import java.util.*;

@Getter
public class BPEVocabInfo extends BaseVocabInfo {
    @Setter
    Map<IntPair, Integer> bpeRanks;

    int wordBoundaryId = -1;
    String wordBoundary;
    Tokenizer.InputMode inputMode;

    public BPEVocabInfo(Tokenizer.InputMode inputMode) {
        this.inputMode = inputMode;
        this.bpeRanks = new HashMap<>();
    }

    public BPEVocabInfo(Map<IntPair, Integer> bpeRanks, Map<String, Integer> vocab, Tokenizer.InputMode inputMode) {
        super(vocab);
        this.bpeRanks = bpeRanks;
        this.inputMode = inputMode;

    }

    public void setWordBoundary(String wordBoundary){
        this.wordBoundary = wordBoundary;
        this.wordBoundaryId = vocab.get(wordBoundary);
    }

    public void setWordBoundaryId(int wordBoundaryId){
        this.wordBoundaryId = wordBoundaryId;
        this.wordBoundary = idToToken.get(wordBoundaryId);
    }


    public void addWordBoundary(String wordBoundaryToken){
        int newId = vocab.size();
        int tokenId = addTokenIfAbsent(wordBoundaryToken);
        if(tokenId != newId){
            throw new IllegalArgumentException("The word boundary token already exists in the vocab.");
        }

        this.setWordBoundary(wordBoundaryToken);
        this.setWordBoundaryId(newId);
    }

    public int getRank(IntPair pair){
        return bpeRanks.getOrDefault(pair, Integer.MAX_VALUE);
    }

    public void nextRank(IntPair pair){
        if(bpeRanks.containsKey(pair)) {
            throw new IllegalArgumentException("The pair already exists in bpeRanks. " + pair.first + ", " + pair.second);
        }

        int newRank = bpeRanks.size();
        bpeRanks.put(pair, newRank); //占位用的无效pair
    }


}
