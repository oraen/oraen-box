package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.util.JSONUtil;
import com.oraen.box.otorch.transformer.Tokenizer;
import com.oraen.box.otorch.transformer.tokenizer.vocab.builder.BPEWordLevelVocabInfoBuilder;
import com.oraen.box.otorch.transformer.tokenizer.vocab.BPEVocabInfo;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class WordLevelBPETokenizer extends BPETokenizer {

    public WordLevelBPETokenizer(BPEVocabInfo BPEVocabInfo) {
        super(BPEVocabInfo);
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


    @Override
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        String wordBoundary = bpeVocabInfo.getWordBoundary();

        for (int id : tokenIds) {
            String token = bpeVocabInfo.getToken(id);

            if (token.equals(wordBoundary)) {
                sb.append(' ');
            } else if (token.startsWith(wordBoundary)) {
                sb.append(' ');
                sb.append(token.substring(wordBoundary.length()));
            } else {
                sb.append(token);
            }
        }

        return sb.toString().trim();
    }

    @Override
    public Tokenizer.InputMode getInputMode() {
        return Tokenizer.InputMode.WORD;
    }

    private List<Integer> encodeWord(String seq) {
        int wordBoundaryId = bpeVocabInfo.getWordBoundaryId();
        int unkId = bpeVocabInfo.getUnkId();
        Map<String, Integer> vocab = bpeVocabInfo.getVocab();

        List<Integer> symbols = new ArrayList<>();
        symbols.add(wordBoundaryId);
        // character-level init
        for (char c : seq.toCharArray()) {
            String s = String.valueOf(c);
            symbols.add(vocab.getOrDefault(s, unkId));
        }

        return bpeMerge(symbols);
    }

    public static void main(String[] args) throws IOException {
        String testFilePath = "E:\\it\\project\\idea\\oraen-box\\oraen-box-otorch\\src\\main\\resources\\corpus\\corpusTest.txt";
  //      String testFilePath = "E:\\it\\project\\idea\\oraen-box\\oraen-box-otorch\\src\\main\\resources\\corpus\\corpus-enTest.txt";
        String content =  new String(Files.readAllBytes(Paths.get(testFilePath)), StandardCharsets.UTF_8);
        BPEVocabInfo bpeVocabInfo = BPEWordLevelVocabInfoBuilder.builder()
                .corpus(content)
                .vocabSize(1000)
                .unk("<|unk|>")
                .wordBoundary("Ġ")
                .build();

        System.out.println("bpeVocabInfo: " + JSONUtil.toJson(bpeVocabInfo));

        WordLevelBPETokenizer tokenizer = new WordLevelBPETokenizer(bpeVocabInfo);

        String text = "我中文名叫阿萨德，英文名叫asd，so，my name is asd，你懂吗";
        System.out.println("source: " + text);

        int[] encode = tokenizer.encode(text);
        System.out.println("encode: " + JSONUtil.toJson(encode));

        String decode = tokenizer.decode(encode);
        System.out.println("decode: " + decode);

    }

}
