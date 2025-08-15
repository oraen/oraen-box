package oraen.box.loader;

import java.util.concurrent.Executor;

public interface LoadContext extends DataLoaderContainer{

    /**
     * Get the result of the data loader.
     *
     * @return the name of the data loader
     */
    <T> T getDataLoadData(String name, Class<T> clazz);

    <T> T getDataLoadData(String name);

    ExecResult getDataLoadResult(String name);

    <T extends Throwable> T getDataLoadError(String name, Class<T> clazz);

    LoadStatus getDataLoadStatus(String name);

    /**
     * Get the initialization parameter of the contex
     *
     * @param clazz the class type of the parameter
     * @return the value of the initialization parameter
     */
    <T> T getInitParam(Class<T> clazz);

    <T> T getInitParam();

    <T> T getResp(Class<T> clazz);

    <T> T getResp();

    /**
     * Get a context variable by its key and class type.
     *
     * @param key the key of the context variable
     * @param clazz the class type of the
     */
    <T> T getContextVariable(String key, Class<T> clazz);

    <T> T getContextVariable(String key);

    /**
     * Set a context variable with a key and value.
     *
     * @param key the key of the context variable
     * @param value the value of the context variable
     */
    void setContextVariable(String key, Object value);

    long getStartTime();

    long getEndTime();

    long getExeTime();

    boolean isSuccess();

    Throwable getThrowable();

    Executor getExecutor();

    void submitTask(Runnable runnable, boolean blockMain);

    enum LoadStatus{
        SUCCESS,
        FALLBACK,
        ERROR,
        ABANDON,
        UNEXECUTED,
    }

}
