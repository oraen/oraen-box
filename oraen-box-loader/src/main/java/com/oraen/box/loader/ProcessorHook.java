package com.oraen.box.loader;

@SuppressWarnings("unchecked")
public interface ProcessorHook<P, R> extends LoaderHook{
    default void beforeExec(String name, LoadContext loadContext){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            beforeNodeExec(name, initParam, result, loadContext);
        }catch (Throwable ignored){

        }

    }

    default void afterExec(String name, LoadContext loadContext, ExecResult loaderExecResult){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            afterNodeExec(name, initParam, result, loadContext, loaderExecResult);
        }catch (Throwable ignored){

        }

    }

    default void beforeLoad(LoadContext loadContext){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            beforeLoad(initParam, result, loadContext);
        }catch (Throwable ignored){

        }

    }

    default void afterLoad(LoadContext loadContext){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            afterLoad(initParam, result, loadContext);
        }catch (Throwable ignored){

        }

    }

    default void onFinalExceptionCaught(Throwable e, LoadContext loadContext){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            onFinalExceptionCaught(initParam, result, e, loadContext);
        }catch (Throwable ignored){

        }


    }

    default void beforeFallback(String name, Throwable e, LoadContext loadContext){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            beforeFallback(name, initParam, result, e, loadContext);
        }catch (Throwable ignored){

        }

    }

    default void afterFallback(String name, Throwable e, LoadContext loadContext, ExecResult loaderExecResult){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            afterFallback(name, initParam, result, e, loadContext, loaderExecResult);
        }catch (Throwable ignored){

        }

    }

    //如果有多个拦截器，优先级为RETRY > GIVE_UP > KEEP
    default RetryCommand onMaybeNeedRetry(String name, Throwable e, int currentRetry, int maxRetry, LoadContext loadContext){
        try{
            P initParam = (P)loadContext.getInitParam(Object.class);
            R result = (R)loadContext.getResp(Object.class);
            return onMaybeNeedRetry(name, initParam, result, e, currentRetry, maxRetry, loadContext);
        }catch (Throwable ignored){
            return RetryCommand.KEEP;
        }
    }

    default void onEveryError(String name, LoadContext context, Throwable e, RunPoi runPoi){
        try{
            P initParam = (P)context.getInitParam(Object.class);
            R result = (R)context.getResp(Object.class);
            onEveryError(name, initParam, result, context, e, runPoi);
        }catch (Throwable ignored){
        }
    }

    //candy
    void beforeNodeExec(String name, P initParam, R result, LoadContext loadContext);

    void afterNodeExec(String name, P initParam, R result, LoadContext loadContext, ExecResult loaderExecResult);

    void beforeLoad(P initParam, R result, LoadContext loadContext);

    void afterLoad(P initParam, R result, LoadContext loadContext);

    void onFinalExceptionCaught(P initParam, R result, Throwable e, LoadContext loadContext);

    void beforeFallback(String name, P initParam, R result, Throwable e, LoadContext loadContext);

    void afterFallback(String name, P initParam, R result, Throwable e, LoadContext loadContext, ExecResult loaderExecResult);

    default RetryCommand onMaybeNeedRetry(String name, P initParam, R result, Throwable e, int currentRetry, int maxRetry, LoadContext loadContext){
        return RetryCommand.KEEP;
    }

    void onEveryError(String name, P initParam, R result, LoadContext context, Throwable e, RunPoi runPoi);
}