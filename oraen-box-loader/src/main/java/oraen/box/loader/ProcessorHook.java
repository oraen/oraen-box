package oraen.box.loader;

import oraen.box.loader.core.CommonLoadContext;

@SuppressWarnings("unchecked")
public interface ProcessorHook<P, R> extends LoaderHook{
    default void beforeExec(String name, LoadContext loadContext){
        P initParam = (P)loadContext.getInitParam(Object.class);
        R result = (R)loadContext.getResp(Object.class);
        beforeNodeExec(name, initParam, result, loadContext);
    }

    default void afterExec(String name, LoadContext loadContext, ExecResult loaderExecResult){
        P initParam = (P)loadContext.getInitParam(Object.class);
        R result = (R)loadContext.getResp(Object.class);
        afterNodeExec(name, initParam, result, loadContext, loaderExecResult);
    }

    default void beforeLoad(LoadContext loadContext){
        P initParam = (P)loadContext.getInitParam(Object.class);
        R result = (R)loadContext.getResp(Object.class);
        beforeLoad(initParam, result, loadContext);

    }

    default void afterLoad(LoadContext loadContext){
        P initParam = (P)loadContext.getInitParam(Object.class);
        R result = (R)loadContext.getResp(Object.class);
        afterLoad(initParam, result, loadContext);
    }


    void beforeNodeExec(String name, P initParam, R result, LoadContext loadContext);

    void afterNodeExec(String name, P initParam, R result, LoadContext loadContext, ExecResult loaderExecResult);

    void beforeLoad(P initParam, R result, LoadContext loadContext);

    void afterLoad(P initParam, R result, LoadContext loadContext);

}