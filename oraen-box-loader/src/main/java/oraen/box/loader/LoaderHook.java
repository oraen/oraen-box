package oraen.box.loader;

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

    default void beforeFallback(String name, Throwable e, LoadContext loadContext){

    }

    default void afterFallback(String name, Throwable e, LoadContext loadContext, ExecResult loaderExecResult){

    }

    //如果有多个拦截器，优先级为RETRY > GIVE_UP > KEEP
    default RetryCommand onMaybeNeedRetry(String name, Throwable e, int currentRetry, int maxRetry, LoadContext loadContext){
        return RetryCommand.KEEP;
    }


    default void onEveryError(String name, LoadContext context, Throwable e, RunPoi runPoi){

    }

    enum RunPoi{
        NORMAL,
        FALLBACK
    }
}