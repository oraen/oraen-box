package oraen.box.loader;

import java.util.List;

public interface DataLoader<T> {

    /**
     * Get the name of this data loader.
     *
     * @return the name of the data loader
     */
    String name();

    /**
     * Get the dependencies of this data loader.
     *
     * @return a list of dependency names
     */
    List<String> dependencies();

    /**
     * Load data based on the provided context.
     *
     * @param context the context containing necessary information for loading data
     * @return the loaded data
     */
    T getData(LoadContext context);

    /**
     * Load data based on the provided context, with a default implementation that checks if loading is needed.
     *
     * @param context the context containing necessary information for loading data
     * @return the loaded data, or null if loading is not needed
     */
    default T fallback(LoadContext context, Throwable e){
        throw e instanceof RuntimeException ? (RuntimeException)e : new RuntimeException(e);
    }

    /**
     * Check if the data needs to be loaded.
     *
     * @param context the context containing necessary information for loading data
     * @return true if data needs to be loaded, false otherwise
     */
    default boolean needLoad(LoadContext context) {
        return true;
    }

    default int maxRetry() {
        return 1;
    }

    //如果返回keep，则由maxRetry决定
    default RetryCommand needRetry(LoadContext context, Throwable e) {
        return RetryCommand.KEEP;
    }


}
