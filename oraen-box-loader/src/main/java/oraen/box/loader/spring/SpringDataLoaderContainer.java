package oraen.box.loader.spring;

import oraen.box.common.util.SpringBeanUtil;
import oraen.box.loader.DataLoader;
import oraen.box.loader.DataLoaderContainer;

import java.util.List;

@SuppressWarnings("all")
public class SpringDataLoaderContainer implements DataLoaderContainer {

    private static SpringDataLoaderContainer singleton = new SpringDataLoaderContainer();

    @Override
    public <T> DataLoader<T> getDataLoader(String name, Class<T> clazz) {
        return (DataLoader<T>)getDataLoader(name);
    }

    @Override
    public DataLoader<?> getDataLoader(String name) {
        return SpringBeanUtil.getBean(name, DataLoader.class);
    }

    @Override
    public List<DataLoader<?>> getDataLoaders() {
        List<DataLoader> rawList = SpringBeanUtil.getBeanList(DataLoader.class);
        // 这里强制转换为 List<DataLoader<?>>，逻辑上是安全的，因为 DataLoader<?> 包含所有泛型参数
        return (List<DataLoader<?>>)(List<?>) rawList;
    }

    public static SpringDataLoaderContainer getSingleton() {
        return singleton;
    }
}
