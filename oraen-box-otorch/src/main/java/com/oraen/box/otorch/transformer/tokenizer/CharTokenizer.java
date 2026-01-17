package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.otorch.transformer.Tokenizer;

import java.util.*;

public class CharTokenizer implements Tokenizer {

    protected final Map<String, Integer> vocab;
    protected final String[] idToToken;

    protected final int padId;
    protected final int bosId;
    protected final int eosId;
    protected final int unkId;

    protected final String unkToken;

    public CharTokenizer(
            Map<String, Integer> vocab,
            int padId,
            int bosId,
            int eosId,
            int unkId,
            String unkToken
    ) {
        this.vocab = vocab;
        this.padId = padId;
        this.bosId = bosId;
        this.eosId = eosId;
        this.unkId = unkId;
        this.unkToken = unkToken;

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
        List<Integer> out = new ArrayList<>();

        for (int i = 0; i < text.length(); i++) {
            String c = String.valueOf(text.charAt(i));
            out.add(vocab.getOrDefault(c, unkId));
        }

        return out.stream().mapToInt(Integer::intValue).toArray();
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
            sb.append(token == null ? unkToken : token);
        }

        return sb.toString();
    }

    // =========================
    // Meta
    // =========================

    @Override public int vocabSize() { return vocab.size(); }
    @Override public int padTokenId() { return padId; }
    @Override public int bosTokenId() { return bosId; }
    @Override public int eosTokenId() { return eosId; }
}
