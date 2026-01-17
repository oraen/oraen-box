package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.otorch.transformer.Tokenizer;

import java.util.*;

public class WordPieceTokenizer implements Tokenizer {

    protected final Map<String, Integer> vocab;
    protected final String[] idToToken;

    protected final int padId;
    protected final int bosId;
    protected final int eosId;
    protected final int unkId;

    protected final String unkToken;
    protected final String continuationPrefix; // usually "##"

    public WordPieceTokenizer(
            Map<String, Integer> vocab,
            int padId,
            int bosId,
            int eosId,
            int unkId,
            String unkToken,
            String continuationPrefix
    ) {
        this.vocab = vocab;
        this.padId = padId;
        this.bosId = bosId;
        this.eosId = eosId;
        this.unkId = unkId;
        this.unkToken = unkToken;
        this.continuationPrefix = continuationPrefix;

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
        List<Integer> out = new ArrayList<>();

        int start = 0;
        int len = word.length();

        while (start < len) {
            int end = len;
            Integer bestId = null;

            while (start < end) {
                String sub = word.substring(start, end);
                if (start > 0) {
                    sub = continuationPrefix + sub;
                }

                bestId = vocab.get(sub);
                if (bestId != null) break;

                end--;
            }

            if (bestId == null) {
                out.add(unkId);
                break;
            }

            out.add(bestId);
            start = end;
        }

        return out;
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

            if (token.startsWith(continuationPrefix)) {
                sb.append(token.substring(continuationPrefix.length()));
            } else {
                if (sb.length() > 0) sb.append(' ');
                sb.append(token);
            }
        }

        return sb.toString();
    }

}
