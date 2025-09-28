package com.oraen.box.loader;

@SuppressWarnings("all")
public interface LoaderHandler {

    <T, U> ExecLog execDataLoadWithLog(DataLoader<T> dataLoader, U initParam, Object initResp);

    default <T, U> ExecResult execDataLoad(DataLoader<T> dataLoader, U initParam, Object initResp){
        ExecLog execLog = execDataLoadWithLog(dataLoader, initParam, initResp);
        if(execLog == null) {
            return null;
        }

        Throwable throwable = execLog.getContext().getThrowable();
        if(throwable != null){
            throw new RuntimeException(throwable);
        }
        return execLog.getContext().getDataLoadResult(dataLoader.name());
    }

    default <T, U> T getData(DataLoader<T> dataLoader, U initParam, Object initResp){
        ExecLog execLog = execDataLoadWithLog(dataLoader, initParam, initResp);
        if(execLog == null) {
            return null;
        }

        Throwable throwable = execLog.getContext().getThrowable();
        if(throwable != null){
            throw new RuntimeException(throwable);
        }

        return (T)execLog.getData();
    }





}
