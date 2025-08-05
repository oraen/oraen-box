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


}