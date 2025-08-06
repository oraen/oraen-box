package oraen.box.loader;

import oraen.box.loader.core.CommonLoadContext;

public interface LoaderHook {
    default void beforeExec(String name, LoadContext loadContext){

    }

    default void afterExec(String name, LoadContext loadContext, ExecResult loaderExecResult){

    }

    default void beforeLoad(LoadContext loadContext){

    }

    default void afterLoad(LoadContext loadContext){

    }


    default void onFinalExceptionCaught(Throwable e, LoadContext loadContext){

    }

    default void onLoaderLoadExceptionCaught(String name, Throwable e, LoadContext loadContext){

    }

    default void beforeRetry(Throwable e, LoadContext loadContext){

    }

    default void afterRetry(Throwable e, LoadContext loadContext){

    }


}