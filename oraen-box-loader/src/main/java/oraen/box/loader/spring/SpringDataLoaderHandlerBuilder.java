//package oraen.box.loader.spring;
//
//import oraen.box.loader.core.CommonLoaderHandler;
//
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//import java.util.concurrent.*;
//
//public class SpringDataLoaderHandlerBuilder {
//
//    private volatile static List<Executor> defaultExecutors = null;
//
//    private final List<Executor> executors = new ArrayList<>();
//
//    private long timeout = 2000L; // 默认2秒
//
//    public static SpringDataLoaderHandlerBuilder builder(){
//        return new SpringDataLoaderHandlerBuilder();
//    }
//
//    public SpringDataLoaderHandlerBuilder executors(List<Executor> executors) {
//        this.executors.addAll(executors);
//        return this;
//    }
//
//    public SpringDataLoaderHandlerBuilder executors(Executor... executors) {
//        this.executors.addAll(Arrays.asList(executors));
//        return this;
//    }
//
//    public SpringDataLoaderHandlerBuilder executor(Executor executor) {
//        this.executors.add(executor);
//        return this;
//    }
//
//    public SpringDataLoaderHandlerBuilder timeout(long timeout) {
//        this.timeout = timeout;
//        return this;
//    }
//
//
//    public CommonLoaderHandler build() {
//        List<Executor> useExecutors = this.executors;
//        if(useExecutors.isEmpty()){
//            if(defaultExecutors == null) {
//                initDefaultExecutors();
//            }
//
//            useExecutors = defaultExecutors;
//        }
//
//        return new CommonLoaderHandler(
//                SpringDataLoaderContainer.getSingleton(),
//                useExecutors,
//                timeout // Default timeout, can be customized later
//        );
//    }
//
//    public synchronized static void initDefaultExecutors() {
//        if (defaultExecutors != null) {
//            return; // 已经初始化过了
//        }
//        defaultExecutors = new ArrayList<>();
//        int executorCount = 4; // 默认线程池数量
//        // 获取 CPU 核心数
//        int cpuCores = Runtime.getRuntime().availableProcessors();
//        int corePoolSize = cpuCores * 4;
//        int maximumPoolSize = cpuCores * 4;
//
//        for(int i = 0; i < executorCount; i++) {
//            ThreadPoolExecutor executor = new ThreadPoolExecutor(
//                    corePoolSize,
//                    maximumPoolSize,
//                    120L,
//                    TimeUnit.SECONDS,
//                    new LinkedBlockingQueue<>(),
//                    Executors.defaultThreadFactory(),
//                    new ThreadPoolExecutor.AbortPolicy()
//            );
//            defaultExecutors.add(executor);
//        }
//
//    }
//}
