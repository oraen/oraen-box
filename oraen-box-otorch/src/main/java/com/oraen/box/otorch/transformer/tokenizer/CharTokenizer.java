package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.otorch.transformer.Tokenizer;

import java.util.*;

public class CharTokenizer implements Tokenizer {

    protected final Map<String, Integer> vocab;
    protected final String[] idToToken;

    protected final int unkId;


    public CharTokenizer(Map<String, Integer> vocab, int unkId) {
        this.vocab = vocab;
        this.unkId = unkId;

        int maxId = Collections.max(vocab.values());
        this.idToToken = new String[maxId + 1];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            idToToken[e.getValue()] = e.getKey();
        }
    }



    @Override
    public int[] encode(String text) {
        List<Integer> out = new ArrayList<>();

        for (int i = 0; i < text.length(); i++) {
            String c = String.valueOf(text.charAt(i));
            out.add(vocab.getOrDefault(c, unkId));
        }

        return out.stream().mapToInt(Integer::intValue).toArray();
    }



    @Override
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();

        for (int id : tokenIds) {
            String token = idToToken[id];
            sb.append(token);
        }

        return sb.toString();
    }

    @Override
    public InputMode getInputMode() {
        return InputMode.CHAR;
    }


}
