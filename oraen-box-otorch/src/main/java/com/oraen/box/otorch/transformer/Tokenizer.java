package com.oraen.box.otorch.transformer;

public interface Tokenizer {
    int[] encode(String text);
    String decode(int[] tokenIds);
}