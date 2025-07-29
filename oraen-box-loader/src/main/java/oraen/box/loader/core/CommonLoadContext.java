package oraen.box.loader.core;

import lombok.Getter;
import lombok.Setter;
import oraen.box.loader.DataLoader;
import oraen.box.loader.ExecResult;
import oraen.box.loader.LoadContext;

import java.util.*;
import java.util.concurrent.*;

@SuppressWarnings("unchecked")
public class CommonLoadContext implements LoadContext {

    @Getter// todo test
    private final Map<String, DataLoader<?>> dataLoaderMap;

    @Getter// todo test
    //初始化时统一写，避免后续并发写操作
    private final Map<String, ExecResult> dataLoadResultMap = new HashMap<>();

    @Getter// todo test
    private final ConcurrentMap<String, Object> contextVariableMap = new ConcurrentHashMap<>();

    @Getter
    private final Object resp;

    @Setter
    Map<String, DataLoadTask> dataLoaderTaskMap;


    @Setter
    @Getter
    Executor executor;

    @Getter
    ConcurrentLinkedQueue<CompletableFuture<?>> extraTasks = new ConcurrentLinkedQueue<>();

    private final Object initParam;

    @Setter
    @Getter
    private long startTime;

    @Setter
    @Getter
    private long endTime;

    @Setter
    @Getter
    private long exeTime;

    @Setter
    @Getter
    private boolean success;

    @Setter
    @Getter
    Throwable throwable;

    public CommonLoadContext(Object initParam, Object initResp, Collection<DataLoader<?>> dataLoaders) {
        this.initParam = initParam;
        this.resp = initResp;
        dataLoaderMap = new HashMap<>();
        for(DataLoader<?> dataLoader : dataLoaders) {
            if (dataLoader != null) {
                dataLoaderMap.put(dataLoader.name(), dataLoader);
            }
        }
    }

    public CommonLoadContext(Object initParam, Object initResp, DataLoader<?>... dataLoaders) {
        this(initParam, initResp, Arrays.asList(dataLoaders));
    }

    public DataLoadTask getDataLoadTask(String name) {
        return dataLoaderTaskMap.get(name);
    }


    @Override
    public <T> DataLoader<T> getDataLoader(String name, Class<T> clazz) {
        return (DataLoader<T>)dataLoaderMap.get(name);
    }

    @Override
    public DataLoader<?> getDataLoader(String name) {
        return dataLoaderMap.get(name);
    }

    @Override
    public List<DataLoader<?>> getDataLoaders(){
        return new ArrayList<>(dataLoaderMap.values());
    }


    @Override
    public <T> T getDataLoadData(String name, Class<T> clazz) {
        return (T)getDataLoadResult(name).getResult();
    }

    @Override
    public <T extends Throwable> T getDataLoadError(String name, Class<T> clazz) {
        return (T)getDataLoadResult(name).getException();
    }

    @Override
    public LoadStatus getDataLoadStatus(String name) {
        int status = getDataLoadResult(name).getStatus();
        if(status == ExecResult.STATUS_SUCCESS) {
            return LoadStatus.SUCCESS;
        } else if(status == ExecResult.STATUS_FALLBACK) {
            return LoadStatus.FALLBACK;
        } else if(status == ExecResult.STATUS_ERROR) {
            return LoadStatus.ERROR;
        } else if(status == ExecResult.STATUS_ABANDON) {
            return LoadStatus.ABANDON;
        } else {
            return LoadStatus.UNEXECUTED;
        }
    }

    public ExecResult getDataLoadResult(String name) {
        ExecResult re = dataLoadResultMap.get(name);
        if(re == null){
            throw new RuntimeException("DataLoader result not found for name: " + name);
        }
        return re;

    }

    public void saveDataLoadResult(String name, ExecResult result) {
        dataLoadResultMap.put(name, result);
    }

    @Override
    public <T> T getInitParam(Class<T> clazz) {
        return (T)initParam;
    }

    @Override
    public <T> T getResp(Class<T> clazz) {
        return (T)resp;
    }

    @Override
    public <T> T getContextVariable(String key, Class<T> clazz) {
        return (T)contextVariableMap.get(key);
    }

    @Override
    public void setContextVariable(String key, Object value) {
        contextVariableMap.put(key, value);
    }

    @Override
    public void submitTask(Runnable runnable) {
        CompletableFuture<Void> future = CompletableFuture.runAsync(runnable, executor);
        extraTasks.add(future);
    }


}
