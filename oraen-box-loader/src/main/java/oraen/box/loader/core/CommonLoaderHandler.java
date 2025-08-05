package oraen.box.loader.core;

import lombok.Getter;
import lombok.Setter;
import oraen.box.common.util.CollectionUtil;
import oraen.box.loader.*;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

@SuppressWarnings("unchecked")
public class CommonLoaderHandler implements LoaderHandler {

    private final DataLoaderContainer dataLoaderContainer;

    private final ConcurrentHashMap<DataLoader<?>,  Map<String, DataLoader<?>> > dataLoaderDependencyCache = new ConcurrentHashMap<>();

    private final List<Executor> executors;

    @Setter
    @Getter
    private long execTimeout;

    @Setter
    @Getter
    private Consumer<ExecLog> afterExec;


    public CommonLoaderHandler(DataLoaderContainer dataLoaderContainer, List<Executor> executors, long execTimeout) {
        this.dataLoaderContainer = dataLoaderContainer;
        if(CollectionUtil.isEmpty(executors)) {
            throw new IllegalArgumentException("Executors must not be null or empty");
        }
        this.executors = new ArrayList<>(executors);
        this.execTimeout = execTimeout;
    }

    @Override
    public <T, U> ExecLog execDataLoadWithLog(DataLoader<T> rootDataLoader, U initParam, Object initResp) {
        Map<String, DataLoader<?>> dataLoaderMap = analyzeNeedLoadDataLoader(rootDataLoader);
        CommonLoadContext loadContext = new CommonLoadContext(initParam, initResp, dataLoaderMap.values());
        Throwable throwable = exec(loadContext);

        loadContext.setSuccess(throwable == null);
        loadContext.setThrowable(throwable);

        ExecLog re = ExecLog.builder()
                .data(loadContext.getDataLoadResult(rootDataLoader.name()).getResult())
                .context(loadContext)
                .build();

        if(afterExec != null) {
            afterExec.accept(re);
        }

        return re;
    }


    private Map<String, DataLoader<?>> analyzeNeedLoadDataLoader(DataLoader<?> rootDataLoader) {
        if(dataLoaderDependencyCache.get(rootDataLoader) != null) {
            return dataLoaderDependencyCache.get(rootDataLoader);
        }

        String hasCircularDependency = LoadUtil.hasCircularDependency(rootDataLoader, dataLoaderContainer);
        if(hasCircularDependency != null) {
            throw new RuntimeException("DataLoader has circular dependency, on " + hasCircularDependency);
        }

        Map<String, DataLoader<?>> re = LoadUtil.getAllDependencies(rootDataLoader, dataLoaderContainer);

        dataLoaderDependencyCache.putIfAbsent(rootDataLoader, re);
        return dataLoaderDependencyCache.get(rootDataLoader);
    }

    private Throwable exec(CommonLoadContext loadContext){
        Executor executor = executors.get(Math.abs(loadContext.hashCode() % executors.size()));
        return LoadLogic.exec(loadContext, executor, execTimeout,null);
    }



}
