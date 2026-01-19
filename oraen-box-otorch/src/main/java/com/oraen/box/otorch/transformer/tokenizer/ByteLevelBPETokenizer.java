package com.oraen.box.otorch.transformer.tokenizer;


import com.oraen.box.common.util.JSONUtil;
import com.oraen.box.otorch.transformer.tokenizer.vocab.builder.BPEByteLevelVocabInfoBuilder;
import com.oraen.box.otorch.transformer.tokenizer.vocab.BPEVocabInfo;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ByteLevelBPETokenizer extends BPETokenizer {


    public ByteLevelBPETokenizer(BPEVocabInfo bpeVocabInfo) {
        super(bpeVocabInfo);
    }

    @Override
    public int[] encode(String text) {

        // 2. byte -> printable chars
        String encoded = ByteEncoder.encode(text);

        // 3. normal BPE encode
        return encodeSequence(encoded).stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public String decode(int[] tokenIds) {

        // 1. 直接拼接 token 字符串（不替换 Ġ！）
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            sb.append(bpeVocabInfo.getToken(id)); // 原样拼接
        }
        String encoded = sb.toString();

        // 2. byte decode
        byte[] bytes = ByteEncoder.decode(encoded);

        // 3. UTF-8 restore
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private List<Integer> encodeSequence(String seq) {
        Map<String, Integer> vocab = bpeVocabInfo.getVocab();
        List<Integer> symbols = new ArrayList<>();

        // character-level init
        for (char c : seq.toCharArray()) {
            String s = String.valueOf(c);
            symbols.add(vocab.get(s));
        }

        return bpeMerge(symbols);
    }

    @Override
    public InputMode getInputMode() {
        return InputMode.BYTE;
    }

    public static void main(String[] args) throws IOException {
        String testFilePath = "E:\\it\\project\\idea\\oraen-box\\oraen-box-otorch\\src\\main\\resources\\corpus\\corpusTest.txt";
        String content =  new String(Files.readAllBytes(Paths.get(testFilePath)), StandardCharsets.UTF_8);
        BPEVocabInfo bpeVocabInfo = BPEByteLevelVocabInfoBuilder.builder()
                .corpus(content)
                .vocabSize(1000)
                .unk("<|unk|>")
                .build();

        System.out.println("===   测试ByteLevelBPETokenizer   ===");
        System.out.println("bpeVocabInfo: " + JSONUtil.toJson(bpeVocabInfo));

        ByteLevelBPETokenizer tokenizer = new ByteLevelBPETokenizer(bpeVocabInfo);

        String text = "我中文名叫阿萨德，英文名叫asd，so，my name is asd，你懂吗";
        System.out.println("source: " + text);
        int[] encode = tokenizer.encode(text);
        System.out.println("encode: " + JSONUtil.toJson(encode));

        String decode = tokenizer.decode(encode);
        System.out.println("decode: " + decode);
    }
}
