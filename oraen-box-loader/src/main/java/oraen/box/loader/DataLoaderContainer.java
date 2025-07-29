package oraen.box.loader;

import java.util.List;

public interface DataLoaderContainer {

    /**
     * Get a DataLoader by its name and class type.
     *
     * @param name the name of the data loader
     * @param clazz the class type of the data loader
     * @param <T> the type of data loaded by the data loader
     * @return the DataLoader instance
     */
    <T> DataLoader<T> getDataLoader(String name, Class<T> clazz);

    DataLoader<?> getDataLoader(String name);

    List<DataLoader<?>> getDataLoaders();


}
