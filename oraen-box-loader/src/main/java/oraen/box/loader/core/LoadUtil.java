package oraen.box.loader.core;

import oraen.box.common.util.ConvertUtil;
import oraen.box.common.util.ParallelUtil;
import oraen.box.common.util.ValueUtil;
import oraen.box.loader.DataLoader;
import oraen.box.loader.DataLoaderContainer;
import oraen.box.loader.LoadContext;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.function.Consumer;
import java.util.function.Function;

public class LoadUtil {


    /**
     * 判断从 rootLoader 开始是否存在循环依赖
     *
     * @param rootLoader 根 DataLoader
     * @param dataLoaderContainer  name -> DataLoader 容器
     * @return 是否存在循环依赖
     */
    public static String hasCircularDependency(DataLoader<?> rootLoader,
                                                DataLoaderContainer dataLoaderContainer) {
        Set<String> visited = new HashSet<>();
        Set<String> recursionStack = new HashSet<>();

        return isCyclic(rootLoader.name(), dataLoaderContainer, visited, recursionStack);
    }

    private static String isCyclic(String name,
                                    DataLoaderContainer dataLoaderContainer,
                                    Set<String> visited,
                                    Set<String> recursionStack) {
        if (recursionStack.contains(name)) {
            // 当前递归路径再次访问到该节点，说明存在环
            return name;
        }

        if (visited.contains(name)) {
            // 已访问且当前路径没有重复，不构成环
            return null;
        }

        visited.add(name);
        recursionStack.add(name);

        DataLoader<?> loader = dataLoaderContainer.getDataLoader(name);
        List<String> dependencies = loader.dependencies();
        if (dependencies != null) {
            for (String dep : dependencies) {
                String isCycle = isCyclic(dep, dataLoaderContainer, visited, recursionStack);
                if (isCycle != null) {
                    return isCycle;
                }
            }
        }

        recursionStack.remove(name);
        return null;
    }

    /**
     * 获取 rootLoader 及其所有递归依赖的 DataLoader
     *
     * @param rootLoader 根 DataLoader
     * @param dataLoaderContainer  全局 name -> DataLoader 映射
     * @return 包含 rootLoader 本身及其依赖的子图 map
     */
    public static Map<String, DataLoader<?>> getAllDependencies(DataLoader<?> rootLoader,
                                                                DataLoaderContainer dataLoaderContainer) {
        Map<String, DataLoader<?>> result = new HashMap<>();
        Set<String> visited = new HashSet<>();
        collectDependencies(rootLoader.name(), dataLoaderContainer, result, visited);
        return result;
    }

    private static void collectDependencies(String name,
                                            DataLoaderContainer dataLoaderContainer,
                                            Map<String, DataLoader<?>> result,
                                            Set<String> visited) {
        if (visited.contains(name)) {
            return;
        }

        visited.add(name);

        DataLoader<?> loader = dataLoaderContainer.getDataLoader(name);
        if (loader != null) {
            result.put(name, loader);
            if(loader.dependencies() != null){
                for (String dep : loader.dependencies()) {
                    collectDependencies(dep, dataLoaderContainer, result, visited);
                }
            }

        }
    }

    public static boolean isAllDependenciesSuccessful(DataLoader<?> dataLoader, LoadContext loadContext){
        for(String name : dataLoader.dependencies()){
            LoadContext.LoadStatus dataLoadStatus = loadContext.getDataLoadStatus(name);
            if(dataLoadStatus != LoadContext.LoadStatus.SUCCESS) {
                //如果有一个依赖没有成功，则返回 false
                return false;
            }
        }
        return true;
    }

    public static boolean isAllDependenciesCompleted(DataLoader<?> dataLoader, LoadContext loadContext){
        for(String name : dataLoader.dependencies()){
            LoadContext.LoadStatus dataLoadStatus = loadContext.getDataLoadStatus(name);
            if(ValueUtil.valueNotIn(dataLoadStatus, LoadContext.LoadStatus.SUCCESS, LoadContext.LoadStatus.FALLBACK)) {
                //如果有一个依赖没有成功，则返回 false
                return false;
            }
        }
        return true;
    }


    public static<T, R> List<R> parallelConvert(Collection<T> sources, Function<T, R> converter, LoadContext loadContext) {
        return ConvertUtil.parallelConvert(sources, converter, loadContext.getExecutor());
    }

    public static<T, R> void parallelExec(Collection<T> sources, Consumer<T> consumer, LoadContext loadContext) {
        ParallelUtil.parallelExec(sources, consumer, loadContext.getExecutor());
    }



}
