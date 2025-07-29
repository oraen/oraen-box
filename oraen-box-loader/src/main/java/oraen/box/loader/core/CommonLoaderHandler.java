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
        long startTime = System.currentTimeMillis();
        Map<String, DataLoader<?>> dataLoaderMap = analyzeNeedLoadDataLoader(rootDataLoader);
        CommonLoadContext loadContext = new CommonLoadContext(initParam, initResp, dataLoaderMap.values());
        loadContext.setStartTime(startTime);
        Throwable throwable = exec(loadContext);
        long endTime = System.currentTimeMillis();
        loadContext.setEndTime(endTime);
        loadContext.setExeTime(endTime - startTime);
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
        loadContext.setExecutor(executor);
        List<DataLoader<?>> dataLoaders = loadContext.getDataLoaders();
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

        //先执行没有依赖的任务
        for(Map.Entry<String, DataLoadTask> entry : dataLoaderTaskMap.entrySet()) {
            DataLoadTask dataLoadTask = entry.getValue();
            //如果没有依赖，直接提交任务
            if(dataLoadTask.isReady()) {
                submitTask(dataLoadTask, loadContext, executor, waitingTasks, mainThread, exceptionRef);
            }
        }

        try {
            long start = System.currentTimeMillis();
            boolean re = waitingTasks.await(execTimeout, TimeUnit.MILLISECONDS);

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
                        extraTasks.get(execTimeout - (System.currentTimeMillis() - start), TimeUnit.MILLISECONDS);
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
        }
        return exceptionRef.get();

    }

    private void submitTask(DataLoadTask dataLoadTask, CommonLoadContext loadContext, Executor executor, CountDownLatch countDownLatch, Thread mainThread, AtomicReference<Throwable> exceptionRef) {
        String name = dataLoadTask.getName();
        DataLoader<?> dataLoader = loadContext.getDataLoader(name);
        if(dataLoader.needLoad(loadContext) && exceptionRef.get() == null) {
            dataLoadTask.setExecutor(() -> {
                ExecResult execResult = loadContext.getDataLoadResult(name);
                execResult.setStatus(ExecResult.STATUS_EXECUTING);
                Object re = null;
                Throwable exception = null;
                int status = ExecResult.STATUS_EXECUTING;
                boolean isSuccess = false;
                boolean useFallback = false;
                boolean isCompleted = false;
                long startTime = System.currentTimeMillis();
                try{
                    re = dataLoader.getData(loadContext);
                    status = ExecResult.STATUS_SUCCESS;
                    isSuccess = true;
                    isCompleted = true;
                }catch (Throwable e) {
                    e.printStackTrace();
                    exception = e;
                    useFallback = true;
                    try{
                        re = dataLoader.fallback(loadContext, e);
                        status = ExecResult.STATUS_FALLBACK;
                        isCompleted = true;
                    }catch (Throwable e1) {
                        status = ExecResult.STATUS_ERROR;
                        exceptionRef.set(e1);
                        //兜底方法异常时代表发生致命错误，中断主线程
                        mainThread.interrupt();
                    }
                }finally {
                    long endTime = System.currentTimeMillis();
                    countDownLatch.countDown();
                    //记录状态
                    execResult.setResult(re);
                    execResult.setException(exception);
                    execResult.setStatus(status);
                    execResult.setExecTime(endTime - startTime);
                    //通知依赖这个任务的任务，并且符合条件时把他们提交到线程池尽可能避免线程切换
                    for(DataLoadTask waitingTask : dataLoadTask.getWaitingTasks()){
                        if(waitingTask.removeDependencyAndIsReady()) {
                            submitTask(waitingTask, loadContext, executor, countDownLatch, mainThread, exceptionRef);
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
                    submitTask(waitingTask, loadContext, executor, countDownLatch, mainThread, exceptionRef);
                }
            }
        }
    }


}
