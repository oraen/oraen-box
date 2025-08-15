package test.oraen.box.loader.loader;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import oraen.box.common.util.ListUtil;
import oraen.box.loader.LoadContext;
import oraen.box.loader.extend.ProcessNode;
import oraen.box.loader.core.CommonLoaderHandler;
import oraen.box.loader.core.CommonMapDataLoaderContainer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;

public class TestLoadersTest2 {


    //pass
    @Test
    public void testHandler() throws Exception {

        MainLoader mainLoader = new MainLoader();
        CommonMapDataLoaderContainer commonMapDataLoaderContainer = new CommonMapDataLoaderContainer()
                .addDataLoader(mainLoader)
                .addDataLoader(new B())
                .addDataLoader(new C())
                .addDataLoader(new D())
                .addDataLoader(new F())
                .addDataLoader(new K());

        InitParam initParam = InitParam.builder()
                .b1(1)
                .b2(2)
                .c1(3)
                .c2(4)
                .f1(5)
                .f2(6)
                .build();


        ArrayList<Executor> defaultExecutors = new ArrayList<>();
        int executorCount = 4; // 默认线程池数量
        // 获取 CPU 核心数
        int cpuCores = Runtime.getRuntime().availableProcessors();
//        int corePoolSize = cpuCores * 4;
        int corePoolSize = 1;
//        int maximumPoolSize = cpuCores * 4;
        int maximumPoolSize = 1;

        for(int i = 0; i < executorCount; i++) {
            ThreadPoolExecutor executor = new ThreadPoolExecutor(
                    corePoolSize,
                    maximumPoolSize,
                    120L,
                    TimeUnit.SECONDS,
                    new LinkedBlockingQueue<>(),
                    Executors.defaultThreadFactory(),
                    new ThreadPoolExecutor.AbortPolicy()
            );
            defaultExecutors.add(executor);
        }

        CommonLoaderHandler commonLoaderHandler = new CommonLoaderHandler(commonMapDataLoaderContainer, defaultExecutors, 2000);

        System.out.println(commonLoaderHandler.execDataLoadWithLog(mainLoader, initParam, new TheResp()).showExeLog());

        Thread.sleep(3000);
    }

    public static class MainLoader implements ProcessNode<InitParam, TheResp> {

        @Override
        public Object process(InitParam param, TheResp resp, LoadContext context) {
            return null;
        }

        @Override
        public String name() {
            return "main";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("k", "b", "c", "f");
        }
    }

    public static class B implements ProcessNode<InitParam, TheResp> {

        @Override
        public Object process(InitParam param, TheResp resp, LoadContext context) {
            return Integer.valueOf(param.b1 + param.b2);
        }

        @Override
        public String name() {
            return "b";
        }

        @Override
        public List<String> dependencies() {
            return Collections.emptyList();
        }
    }

    public static class C implements ProcessNode<InitParam, TheResp> {

        @Override
        public Object process(InitParam param, TheResp resp, LoadContext context) {
            return param.c1 + param.c2;
        }

        @Override
        public String name() {
            return "c";
        }

        @Override
        public List<String> dependencies() {
            return null;
        }
    }

    public static class F implements ProcessNode<InitParam, TheResp> {

        @Override
        public Object process(InitParam param, TheResp resp, LoadContext context) {
            return param.f1 * param.f2;
        }

        @Override
        public String name() {
            return "f";
        }

        @Override
        public List<String> dependencies() {
            return null;
        }
    }

    public static class D implements ProcessNode<InitParam, TheResp> {

        @Override
        public Object process(InitParam param, TheResp resp, LoadContext context) {
            return context.getDataLoadData("b", Integer.class) + context.getDataLoadData("c", Integer.class);
        }

        @Override
        public String name() {
            return "d";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("b", "c");
        }
    }

    public static class K implements ProcessNode<InitParam, TheResp> {

        @Override
        public Object process(InitParam param, TheResp resp, LoadContext context) {
            return context.getDataLoadData("d", Integer.class) / context.getDataLoadData("f", Integer.class);
        }

        @Override
        public String name() {
            return "k";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("d", "f");
        }
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class InitParam{
        Integer b1;
        Integer b2;

        Integer c1;
        Integer c2;

        Integer f1;
        Integer f2;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TheResp{
        Integer resp;
    }






}
