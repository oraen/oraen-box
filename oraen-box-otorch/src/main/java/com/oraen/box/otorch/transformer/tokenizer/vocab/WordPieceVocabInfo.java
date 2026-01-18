package com.oraen.box.otorch.transformer.tokenizer.vocab;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

public class WordPieceVocabInfo extends BaseVocabInfo{

    public final static String DEFAULT_CONTINUATION_PREFIX = "##";

    // usually "##"
    @Getter
    @Setter
    protected String continuationPrefix;

    public WordPieceVocabInfo(){
        this(DEFAULT_CONTINUATION_PREFIX);
    }

    public WordPieceVocabInfo(String continuationPrefix){
        super();
        this.continuationPrefix = continuationPrefix;
    }


}
