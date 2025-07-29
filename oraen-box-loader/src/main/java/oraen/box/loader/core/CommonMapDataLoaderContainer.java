package oraen.box.loader.core;

import oraen.box.loader.DataLoader;
import oraen.box.loader.DataLoaderContainer;

import java.util.*;

@SuppressWarnings("unchecked")
public class CommonMapDataLoaderContainer implements DataLoaderContainer {

    Map<String, DataLoader<?>> dataLoaderMap = new HashMap<>();

    @Override
    public <T> DataLoader<T> getDataLoader(String name, Class<T> clazz) {
        return (DataLoader<T>)dataLoaderMap.get(name);
    }

    @Override
    public DataLoader<?> getDataLoader(String name) {
        return dataLoaderMap.get(name);
    }

    @Override
    public List<DataLoader<?>> getDataLoaders() {
        return new ArrayList<>(dataLoaderMap.values());
    }

    public CommonMapDataLoaderContainer addDataLoader(DataLoader<?> dataLoader) {
        if (dataLoader != null) {
            dataLoaderMap.put(dataLoader.name(), dataLoader);
        }
        return this;
    }

    public CommonMapDataLoaderContainer addDataLoader(DataLoader<?>... dataLoaders) {
        for (DataLoader<?> dataLoader : dataLoaders) {
            addDataLoader(dataLoader);
        }
        return this;
    }
}
