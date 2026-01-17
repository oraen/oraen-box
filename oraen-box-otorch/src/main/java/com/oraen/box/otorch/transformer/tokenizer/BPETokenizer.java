package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.structure.IntPair;
import com.oraen.box.otorch.transformer.Tokenizer;

import java.util.*;

/**
 * Base BPE Tokenizer (word-level with word boundary marker)
 */
public class BPETokenizer implements Tokenizer {

    /** (tokenIdA, tokenIdB) -> merge rank (lower = higher priority) */
    protected final Map<IntPair, Integer> bpeRanks;

    /** token string -> token id */
    protected final Map<String, Integer> vocab;

    /** token id -> token string */
    protected final String[] idToToken;

    protected final int padId;
    protected final int bosId;
    protected final int eosId;
    protected final int unkId;

    protected final String unkToken;
    protected final String wordBoundary;

    public BPETokenizer(
            Map<IntPair, Integer> bpeRanks,
            Map<String, Integer> vocab,
            int padId,
            int bosId,
            int eosId,
            int unkId,
            String unkToken,
            String wordBoundary
    ) {
        this.bpeRanks = bpeRanks;
        this.vocab = vocab;
        this.padId = padId;
        this.bosId = bosId;
        this.eosId = eosId;
        this.unkId = unkId;
        this.unkToken = unkToken;
        this.wordBoundary = wordBoundary;

        int maxId = Collections.max(vocab.values());
        this.idToToken = new String[maxId + 1];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            idToToken[e.getValue()] = e.getKey();
        }
    }

    // =========================
    // Encode
    // =========================

    @Override
    public int[] encode(String text) {
        List<Integer> output = new ArrayList<>();

        int i = 0;
        int n = text.length();

        while (i < n) {
            // skip whitespace
            while (i < n && Character.isWhitespace(text.charAt(i))) {
                i++;
            }
            if (i >= n) break;

            int start = i;
            while (i < n && !Character.isWhitespace(text.charAt(i))) {
                i++;
            }

            String word = text.substring(start, i);
            output.addAll(encodeWord(word));
        }

        return output.stream().mapToInt(Integer::intValue).toArray();
    }

    protected List<Integer> encodeWord(String word) {
        List<Integer> symbols = new ArrayList<>();

        // word boundary as a standalone symbol
        Integer wbId = vocab.get(wordBoundary);
        if (wbId == null) {
            throw new IllegalStateException("wordBoundary token not in vocab");
        }
        symbols.add(wbId);

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
        while (true) {
            IntPair bestPair = null;
            int bestRank = Integer.MAX_VALUE;

            for (int i = 0; i < symbols.size() - 1; i++) {
                IntPair pair = new IntPair(symbols.get(i), symbols.get(i + 1));
                Integer rank = bpeRanks.get(pair);
                if (rank == null) continue;

                String merged = idToToken[pair.first] + idToToken[pair.second];
                Integer mergedId = vocab.get(merged);
                if (mergedId == null) continue;

                if (rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                }
            }

            if (bestPair == null) break;

            List<Integer> newSymbols = new ArrayList<>();
            for (int i = 0; i < symbols.size(); i++) {
                if (i < symbols.size() - 1 &&
                        symbols.get(i) == bestPair.first &&
                        symbols.get(i + 1) == bestPair.second) {

                    String merged = idToToken[bestPair.first] + idToToken[bestPair.second];
                    newSymbols.add(vocab.get(merged));
                    i++;
                } else {
                    newSymbols.add(symbols.get(i));
                }
            }
            symbols = newSymbols;
        }
        return symbols;
    }

    // =========================
    // Decode
    // =========================

    @Override
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();

        for (int id : tokenIds) {
            if (id < 0 || id >= idToToken.length) {
                sb.append(unkToken);
                continue;
            }

            String token = idToToken[id];
            if (token == null) {
                sb.append(unkToken);
                continue;
            }

            if (token.equals(wordBoundary)) {
                sb.append(' ');
            } else {
                sb.append(token);
            }
        }

        return sb.toString().trim();
    }

    // =========================
    // Meta
    // =========================

    @Override public int vocabSize() { return vocab.size(); }
    @Override public int padTokenId() { return padId; }
    @Override public int bosTokenId() { return bosId; }
    @Override public int eosTokenId() { return eosId; }
}
