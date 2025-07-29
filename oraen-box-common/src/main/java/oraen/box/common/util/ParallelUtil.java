package oraen.box.common.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

public class ParallelUtil {

    public static ParallelTask parallelTask(Executor executor) {
        return new ParallelTask(executor);
    }

    public static<T> void parallelExec(Collection<T> collection, Consumer<T> consumer, Executor executor) {
        List<CompletableFuture<Void>> futures = new ArrayList<>(collection.size());
        for (T item : collection) {
            CompletableFuture<Void> voidCompletableFuture = CompletableFuture.runAsync(() -> consumer.accept(item), executor);
            futures.add(voidCompletableFuture);
        }

        for(CompletableFuture<Void> future : futures) {
            future.join(); // 等待所有任务完成
        }
    }



    public static class ParallelTask{
        List<Runnable> runnable = new ArrayList<>();

        final Executor executor;

        public ParallelTask(Executor executor) {
            this.executor = executor;
        }

        public ParallelTask add(Runnable runnable) {
            this.runnable.add(runnable);
            return this;
        }


        public ParallelTask addAll(Runnable... runnables) {
            this.runnable.addAll(Arrays.asList(runnables));
            return this;
        }

        public ParallelTask addAll(Collection<Runnable> runnables) {
            this.runnable.addAll(runnables);
            return this;
        }

        public void runAndWait() {
            List<CompletableFuture<Void>> futures = new ArrayList<>(runnable.size());
            for (Runnable r : runnable) {
                CompletableFuture<Void> voidCompletableFuture = CompletableFuture.runAsync(r, executor);
                futures.add(voidCompletableFuture);
            }

            for(CompletableFuture<Void> future : futures) {
                future.join(); // 等待所有任务完成
            }
        }


    }
}
