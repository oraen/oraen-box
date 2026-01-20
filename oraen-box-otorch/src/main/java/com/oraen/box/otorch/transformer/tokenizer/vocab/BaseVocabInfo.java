package com.oraen.box.otorch.transformer.tokenizer.vocab;

import lombok.Getter;

import java.util.*;

@Getter
public class BaseVocabInfo {
    Map<String, Integer> vocab ;
    List<String> idToToken ;
    int unkId = -1;
    String unk;

    public BaseVocabInfo() {
        this.vocab = new HashMap<>();
        this.idToToken = new ArrayList<>(30000);
    }

    public BaseVocabInfo(Map<String, Integer> vocab){
        this.vocab = vocab;
        int maxId = Collections.max(vocab.values());
        String[] idToTokenArray = new String[maxId + 1];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            idToTokenArray[e.getValue()] = e.getKey();
        }

        this.idToToken = new ArrayList<>(Arrays.asList(idToTokenArray));
    }

    public void setUnk(String unk) {
        this.unk = unk;
        this.unkId = vocab.get(unk);
    }

    public void setUnkId(int unkId) {
        this.unkId = unkId;
        this.unk = idToToken.get(unkId);
    }

    public int getVocabSize(){
        return vocab.size();
    }

    public int getId(String token){
        return vocab.getOrDefault(token, unkId);
    }

    public String getToken(int id){
        return idToToken.get(id);
    }

    public int addTokenIfAbsent(String token){
        if(vocab.containsKey(token)) return vocab.get(token);
        int newId = vocab.size();
        vocab.put(token, newId);
        idToToken.add(token);
        return newId;
    }

    public void addUnk(String unkToken){
        int newId = vocab.size();
        int tokenId = addTokenIfAbsent(unkToken);
        if(tokenId != newId){
            throw new IllegalArgumentException("The unk token already exists in the vocab.");
        }

        this.setUnk(unkToken);
        this.setUnkId(newId);
    }

    public boolean containsToken(String token){
        return vocab.containsKey(token);
    }
}
