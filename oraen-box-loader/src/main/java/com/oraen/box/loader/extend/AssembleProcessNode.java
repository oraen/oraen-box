package com.oraen.box.loader.extend;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.oraen.box.common.function.QuadrupleFunction;
import com.oraen.box.common.function.TripleFunction;
import com.oraen.box.loader.LoadContext;
import com.oraen.box.loader.RetryCommand;

import java.util.List;
import java.util.function.Predicate;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AssembleProcessNode<P, R> implements ProcessNode<P, R>{
    private String name;

    private List<String> dependencies;

    private TripleFunction<P, R, LoadContext, Object> process;

    private QuadrupleFunction<P, R, Throwable, LoadContext, Object> fallback;

    private Predicate<LoadContext> needLoad;

    private Integer maxRetry;

    private QuadrupleFunction<P, R, LoadContext, Throwable, RetryCommand> needRetry;

    public AssembleProcessNode(String name, List<String> dependencies, TripleFunction<P, R, LoadContext, Object> process){
        this.name = name;
        this.dependencies = dependencies;
        this.process = process;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public List<String> dependencies() {
        return dependencies;
    }

    @Override
    public Object process(P param, R resp, LoadContext context) {
        return process.apply(param, resp, context);
    }

    @Override
    public Object fallback(P param, R resp, Throwable t, LoadContext context){
        if(fallback == null){
            throw t instanceof RuntimeException ? (RuntimeException)t : new RuntimeException(t);
        }

        return fallback.apply(param, resp, t, context);
    }

    @Override
    public boolean needLoad(LoadContext context) {
        return needLoad == null || needLoad.test(context);
    }

    @Override
    public int maxRetry() {
        return maxRetry == null || maxRetry < 1 ? 1 : maxRetry;
    }

    @Override
    public RetryCommand needRetry(P param, R resp, LoadContext context, Throwable e) {
        return needRetry == null ? RetryCommand.KEEP : needRetry.apply(param, resp, context, e);
    }

}
