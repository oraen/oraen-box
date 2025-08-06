package oraen.box.loader.core;

import oraen.box.loader.DataLoader;
import oraen.box.loader.ExecResult;
import oraen.box.loader.LoaderHook;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

public class LoadLogic {

    static private  final List<LoaderHook> EMPTY_HOOKS = new ArrayList<>();

    public static Throwable exec(CommonLoadContext loadContext, Executor executor, long execTimeout, Collection<? extends LoaderHook> hooks) {
        if(hooks == null) {
            hooks = EMPTY_HOOKS;
        }

        long startTime = System.currentTimeMillis();
        loadContext.setExecutor(executor);
        List<DataLoader<?>> dataLoaders = loadContext.getDataLoaders();
        //准备提交的任务
        Map<String, DataLoadTask> dataLoaderTaskMap = new HashMap<>(dataLoaders.size());
        CountDownLatch waitingTasks = new CountDownLatch(dataLoaders.size());
        AtomicReference<Throwable> exceptionRef = new AtomicReference<>(null);
        //主线程
        Thread mainThread = Thread.currentThread();

        //初始化任务和环境
        for (DataLoader<?> dataLoader : dataLoaders) {
            String name = dataLoader.name();
            ExecResult result = ExecResult.builder()
                    .status(ExecResult.STATUS_INIT)
                    .build();
            loadContext.saveDataLoadResult(name, result);
            DataLoadTask task = new DataLoadTask(name);
            DataLoadTask old = dataLoaderTaskMap.put(name, task);
            if(old != null) {
                throw new IllegalStateException("DataLoader name must be unique, but found duplicate: " + name);
            }
        }

        loadContext.setDataLoaderTaskMap(dataLoaderTaskMap);

        //构建任务之间的依赖关系，并且准备执行
        for (DataLoader<?> dataLoader : dataLoaders) {
            String name = dataLoader.name();
            DataLoadTask dataLoadTask = dataLoaderTaskMap.get(name);
            List<String> dependencies = dataLoader.dependencies();
            if(dependencies != null) {
                dependencies.forEach(dependency -> {
                    DataLoadTask dependencyTask = dataLoaderTaskMap.get(dependency);
                    dataLoadTask.addDependency(dependencyTask);
                });
            }

            ExecResult execResult = loadContext.getDataLoadResult(name);
            execResult.setStatus(ExecResult.STATUS_WAITING);
        }

        List<LoaderHook> reverseHooks = new ArrayList<>(hooks);
        Collections.reverse(reverseHooks);
        for(LoaderHook hook : hooks) {
            hook.beforeLoad(loadContext);
        }

        //先执行没有依赖的任务
        for(Map.Entry<String, DataLoadTask> entry : dataLoaderTaskMap.entrySet()) {
            DataLoadTask dataLoadTask = entry.getValue();
            //如果没有依赖，直接提交任务
            if(dataLoadTask.isReady()) {
                submitTask(dataLoadTask, loadContext, executor, waitingTasks, mainThread, exceptionRef, hooks, reverseHooks);
            }
        }

        try {
            boolean re = waitingTasks.await(execTimeout - (System.currentTimeMillis() - startTime), TimeUnit.MILLISECONDS);

            if(! re){
                TimeoutException timeoutException = new TimeoutException("DataLoader execution timed out after " + execTimeout + " milliseconds");
                exceptionRef.set(timeoutException);
            }

            boolean allExtraTaskDone = false;
            int checkCount = 2;
            //如果有额外的任务需要执行，并且需要处理额外任务提交其他额外任务的情况
            while(!allExtraTaskDone || checkCount > 0) {
                //等待所有任务执行完毕
                allExtraTaskDone = true;
                for(CompletableFuture<?> extraTasks : loadContext.getExtraTasks()){
                    if(! extraTasks.isDone()){
                        allExtraTaskDone = false;
                        checkCount = 2;
                        extraTasks.get(execTimeout - (System.currentTimeMillis() - startTime), TimeUnit.MILLISECONDS);
                    }
                }

                if(allExtraTaskDone){
                    checkCount --;
                }
            }

        } catch (InterruptedException e) {
            if(exceptionRef.get() == null){
                exceptionRef.set(e);
            }
        }catch (Exception e){
            exceptionRef.set(e);
        }finally {
            Throwable throwable = exceptionRef.get();
            long endTime = System.currentTimeMillis();
            loadContext.setEndTime(endTime);
            loadContext.setStartTime(startTime);
            loadContext.setExeTime(endTime - startTime);
            loadContext.setThrowable(throwable);
            loadContext.setSuccess(throwable == null);

            for(LoaderHook hook : reverseHooks) {
                if(throwable != null){
                    hook.onFinalExceptionCaught(throwable, loadContext);
                }

                hook.afterLoad(loadContext);
            }

        }
        return exceptionRef.get();

    }

    private static void submitTask(DataLoadTask dataLoadTask, CommonLoadContext loadContext, Executor executor,
                                   CountDownLatch countDownLatch, Thread mainThread, AtomicReference<Throwable> exceptionRef,
                                   Collection<? extends LoaderHook> hooks, Collection<? extends LoaderHook> reverseHooks) {
        String name = dataLoadTask.getName();
        DataLoader<?> dataLoader = loadContext.getDataLoader(name);
        if(dataLoader.needLoad(loadContext) && exceptionRef.get() == null) {
            dataLoadTask.setExecutor(() -> {
                ExecResult execResult = loadContext.getDataLoadResult(name);
                execResult.setStatus(ExecResult.STATUS_EXECUTING);
                long startTime = System.currentTimeMillis();
                try{
                    //钩子函数
                    for(LoaderHook hook : hooks) {
                        hook.beforeExec(dataLoader.name(), loadContext);
                    }

                    Object re = dataLoader.getData(loadContext);
                    execResult.setResult(re);
                    execResult.setStatus(ExecResult.STATUS_SUCCESS);

                    execResult.setSuccess(true);
                }catch (Throwable e) {
                    execResult.setException(e);
                    execResult.setUseFallback(true);
                    try{
                        for(LoaderHook hook : hooks) {
                            hook.onLoaderLoadExceptionCaught(dataLoader.name(), e, loadContext);
                        }

                        Object re = dataLoader.fallback(loadContext, e);
                        execResult.setResult(re);
                        execResult.setStatus(ExecResult.STATUS_FALLBACK);
                    }catch (Throwable e1) {
                        execResult.setStatus(ExecResult.STATUS_ERROR);

                        execResult.setException(e1);
                        exceptionRef.set(e1);
                        //兜底方法异常时代表发生致命错误，中断主线程
                        mainThread.interrupt();
                    }
                }finally {
                    long endTime = System.currentTimeMillis();
                    countDownLatch.countDown();
                    execResult.setCompleted(true);
                    execResult.setExecTime(endTime - startTime);

                    //钩子函数
                    for(LoaderHook hook : reverseHooks) {
                        hook.afterExec(dataLoader.name(), loadContext, execResult);
                    }
                    //通知依赖这个任务的任务，并且符合条件时把他们提交到线程池尽可能避免线程切换
                    for(DataLoadTask waitingTask : dataLoadTask.getWaitingTasks()){
                        if(waitingTask.removeDependencyAndIsReady()) {
                            submitTask(waitingTask, loadContext, executor, countDownLatch, mainThread, exceptionRef, hooks, reverseHooks);
                        }
                    }

                }
            });
            executor.execute(dataLoadTask.getExecutor());
        } else {
            countDownLatch.countDown();
            ExecResult execResult = loadContext.getDataLoadResult(name);
            execResult.setStatus(ExecResult.STATUS_ABANDON);
            //这个任务不需要执行，也需要提醒依赖这个服务的任务执行
            for(DataLoadTask waitingTask : dataLoadTask.getWaitingTasks()){
                if(waitingTask.removeDependencyAndIsReady()) {
                    submitTask(waitingTask, loadContext, executor, countDownLatch, mainThread, exceptionRef, hooks, reverseHooks);
                }
            }
        }
    }
}
