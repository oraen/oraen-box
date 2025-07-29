package oraen.box.loader.core;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import oraen.box.loader.DataLoader;
import oraen.box.loader.LoadContext;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AssembleDataLoader<T> implements DataLoader<T> {

    private String name;

    private List<String> dependencies;

    private Function<LoadContext, T> getDataFunction;


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
        return getDataFunction.apply(context);
    }
}
