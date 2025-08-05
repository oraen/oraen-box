package oraen.box.loader.extend;

import oraen.box.common.util.CollectionUtil;
import oraen.box.loader.DataLoader;
import oraen.box.loader.LoaderHook;
import oraen.box.loader.ProcessNode;
import oraen.box.loader.core.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

@SuppressWarnings("unchecked")
public class ParallelDataBuilder<P, R> {

    public static volatile List<Executor> defaultExecutors = null;

    private final List<ProcessNode<P, R>> processNodes = new ArrayList<>();

    private long execTimeout = 2000L;

    private List<Executor> executors;

    private List<LoaderHook> hooks = new ArrayList<>();

    public static<P, R> ParallelDataBuilder<P, R> builder() {
        return new ParallelDataBuilder<>();
    }

    private static List<Executor> getDefaultExecutors(){
        if(defaultExecutors == null) {
            synchronized (ParallelDataBuilder.class) {
                if(defaultExecutors == null) {
                    defaultExecutors = new ArrayList<>();
                    int executorCount = 4; // 默认线程池数量
                    // 获取 CPU 核心数
                    int cpuCores = Runtime.getRuntime().availableProcessors();
                    int corePoolSize = cpuCores * 4;
                    int maximumPoolSize = cpuCores * 4;

                    for(int i = 0; i < executorCount; i++) {
                        ThreadPoolExecutor executor = new ThreadPoolExecutor(
                                corePoolSize,
                                maximumPoolSize,
                                120L,
                                TimeUnit.SECONDS,
                                new LinkedBlockingQueue<>(),
                                Executors.defaultThreadFactory(),
                                new ThreadPoolExecutor.CallerRunsPolicy()
                        );
                        defaultExecutors.add(executor);
                    }
                }
            }
        }

        return defaultExecutors;
    }

    public ParallelDataBuilder<P, R> addNodes(ProcessNode<P, R>... processNode) {
        return addNodes(Arrays.asList(processNode));
    }

    public ParallelDataBuilder<P, R> addNodes(List<ProcessNode<P, R>> processNodes) {
        this.processNodes.addAll(processNodes);
        return this;
    }

    public ParallelDataBuilder<P, R> setExecTimeout(long execTimeout) {
        this.execTimeout = execTimeout;
        return this;
    }

    public ParallelDataBuilder<P, R> setExecutors(List<Executor> executors) {
        this.executors = new ArrayList<>(executors);
        return this;
    }

    public ParallelDataBuilder<P, R> addExecutors(List<Executor> executors) {
        this.executors.addAll(executors);
        return this;
    }


    public ParallelDataBuilder<P, R> addExecutors(Executor... executors) {
        this.executors.addAll(Arrays.asList(executors));
        return this;
    }

    public ParallelDataBuilder<P, R> setHooks(List<? extends LoaderHook> hooks) {
        this.hooks = new ArrayList<>(hooks);
        return this;
    }

    public ParallelDataBuilder<P, R> addHooks(List<? extends LoaderHook> hooks) {
        this.hooks.addAll(hooks);
        return this;
    }


    public ParallelDataBuilder<P, R> addHooks(LoaderHook... executors) {
        return addHooks(Arrays.asList(executors));
    }

    public ParallelDataBuilder<P, R> ensure(){
        CommonMapDataLoaderContainer commonMapDataLoaderContainer = new CommonMapDataLoaderContainer();
        commonMapDataLoaderContainer.addDataLoaders(processNodes);
        AssembleDataLoader<Object> rootDataLoader = AssembleDataLoader.builder()
                .dataFunction(loadContext -> null)
                .name("_oraen_temp_root")
                .dependencies(processNodes.stream().map(DataLoader::name).collect(Collectors.toList()))
                .build();

        commonMapDataLoaderContainer.addDataLoader(rootDataLoader);

        String hasCircularDependency = LoadUtil.hasCircularDependency(rootDataLoader, commonMapDataLoaderContainer);
        if(hasCircularDependency != null) {
            throw new RuntimeException("DataLoader has circular dependency, on " + hasCircularDependency);
        }

        return this;
    }

    public R buildResp(P initParam, R initResp) {
        CommonLoadContext commonLoadContext = buildRespWithDetail(initParam, initResp);
        if(commonLoadContext.getThrowable() != null) {
            throw new RuntimeException("Data loading failed", commonLoadContext.getThrowable());
        }

        return (R) commonLoadContext.getResp();
    }

    public R buildResp(Supplier<P> initParamSupplier, Supplier<R> initRespSupplier) {
        return buildResp(initParamSupplier.get(), initRespSupplier.get());
    }

    public CommonLoadContext buildRespWithDetail(P initParam, R initResp) {
        CommonLoadContext loadContext = new CommonLoadContext(initParam, initResp, processNodes);
        List<Executor> executors = this.executors;
        if(CollectionUtil.isEmpty(executors)) {
            executors = getDefaultExecutors();
        }

        Executor executor = executors.get(Math.abs(loadContext.hashCode() % executors.size()));
        LoadLogic.exec(loadContext, executor, execTimeout, hooks);
        return loadContext;
    }

    public CommonLoadContext buildRespWithDetail(Supplier<P> initParamSupplier, Supplier<R> initRespSupplier) {
        P initParam = initParamSupplier.get();
        R initResp = initRespSupplier.get();
        return buildRespWithDetail(initParam, initResp);
    }



}
