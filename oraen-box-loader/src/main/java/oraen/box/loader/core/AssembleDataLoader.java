package oraen.box.loader.core;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import oraen.box.loader.DataLoader;
import oraen.box.loader.LoadContext;
import oraen.box.loader.RetryCommand;

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AssembleDataLoader<T> implements DataLoader<T> {

    private String name;

    private List<String> dependencies;

    private Function<LoadContext, T> dataFunction;

    private BiFunction<LoadContext, Throwable, T> fallback;

    private Predicate<LoadContext> needLoad;

    private Integer maxRetry;

    private BiFunction<LoadContext, Throwable, RetryCommand> needRetry;

    public AssembleDataLoader(String name, List<String> dependencies, Function<LoadContext, T> dataFunction){
        this.name = name;
        this.dependencies = dependencies;
        this.dataFunction = dataFunction;
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
    public T getData(LoadContext context) {
        return dataFunction.apply(context);
    }

    @Override
    public T fallback(LoadContext context, Throwable e){
        if(fallback == null){
            throw e instanceof RuntimeException ? (RuntimeException)e : new RuntimeException(e);
        }

        return fallback.apply(context, e);
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
    public RetryCommand needRetry(LoadContext context, Throwable e) {
        return needRetry == null ? RetryCommand.KEEP : needRetry.apply(context, e);
    }


}
