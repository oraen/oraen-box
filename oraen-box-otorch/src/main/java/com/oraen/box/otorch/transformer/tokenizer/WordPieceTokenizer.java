package com.oraen.box.otorch.transformer.tokenizer;

import com.oraen.box.common.util.JSONUtil;
import com.oraen.box.otorch.transformer.Tokenizer;
import com.oraen.box.otorch.transformer.tokenizer.vocab.BPEVocabInfo;
import com.oraen.box.otorch.transformer.tokenizer.vocab.WordPieceVocabInfo;
import com.oraen.box.otorch.transformer.tokenizer.vocab.builder.BPEWordLevelVocabInfoBuilder;
import com.oraen.box.otorch.transformer.tokenizer.vocab.builder.WordPieceVocabInfoBuilder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class WordPieceTokenizer implements Tokenizer {

    WordPieceVocabInfo wordPieceVocabInfo;

    public WordPieceTokenizer(WordPieceVocabInfo wordPieceVocabInfo) {
        this.wordPieceVocabInfo = wordPieceVocabInfo;
    }



    @Override
    public int[] encode(String text) {
        List<Integer> output = new ArrayList<>();

        // 按空白切词
        String[] words = text.split("\\s+");

        for (String word : words) {
            if (!word.isEmpty()) {
                output.addAll(encodeWord(word));
            }
        }

        return output.stream().mapToInt(Integer::intValue).toArray();
    }

    protected List<Integer> encodeWord(String word) {
        List<Integer> out = new ArrayList<>();
        String continuationPrefix = wordPieceVocabInfo.getContinuationPrefix();
        int unkId = wordPieceVocabInfo.getUnkId();

        int start = 0;
        int len = word.length();

        while (start < len) {
            int end = len;
            int bestId = unkId;

            while (start < end) {
                String sub = word.substring(start, end);
                if (start > 0) {
                    sub = continuationPrefix + sub;
                }

                bestId = wordPieceVocabInfo.getId(sub);
                if (bestId != unkId) break;
                end--;
            }

            out.add(bestId);
            if (bestId == unkId) {
                break;
            }

            start = end;
        }

        return out;
    }


    @Override
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        String continuationPrefix = wordPieceVocabInfo.getContinuationPrefix();

        for (int id : tokenIds) {
            //忽略出现异常id的情况，肯定是使用出了问题
            String token = wordPieceVocabInfo.getToken(id);

            if (token.startsWith(continuationPrefix)) {
                sb.append(token.substring(continuationPrefix.length()));
            } else {
                sb.append(' ');
                sb.append(token);
            }
        }

        return sb.toString().trim();
    }

    @Override
    public InputMode getInputMode() {
        return InputMode.WORD;
    }


    public static void main(String[] args) throws IOException {
       // String testFilePath = "E:\\it\\project\\idea\\oraen-box\\oraen-box-otorch\\src\\main\\resources\\corpus\\corpus-enTest.txt";
        String testFilePath = "/Users/corki/IdeaProjects/ad/oraen-box/oraen-box-otorch/src/main/resources/corpus/corpus-enTest.txt";
        String content =  new String(Files.readAllBytes(Paths.get(testFilePath)), StandardCharsets.UTF_8);
        WordPieceVocabInfo wordpieceVocabInfo = WordPieceVocabInfoBuilder.builder()
                .corpus(content)
                .vocabSize(1000)
                .highFreqThreshold(10)
                .unk("<|unk|>")
                .build();

        System.out.println("wordpiece VocabInfo: " + JSONUtil.toJson(wordpieceVocabInfo));

        WordPieceTokenizer tokenizer = new WordPieceTokenizer(wordpieceVocabInfo);

        String text = "im corki not asd, do you know?";
        System.out.println("source: " + text);

        int[] encode = tokenizer.encode(text);
        System.out.println("encode: " + JSONUtil.toJson(encode));

        String decode = tokenizer.decode(encode);
        System.out.println("decode: " + decode);

    }

}
