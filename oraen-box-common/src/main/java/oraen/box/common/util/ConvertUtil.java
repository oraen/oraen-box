package oraen.box.common.util;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.function.Function;

public class ConvertUtil {

    public static<T, R> List<R> parallelConvert(Collection<T> sources, Function<T, R> converter, Executor executor) {
        List<R> results = new ArrayList<>(Collections.nCopies(sources.size(), null));
        CountDownLatch latch = new CountDownLatch(sources.size());
        int i = 0;
        for(T source : sources) {
            final int index = i;
            executor.execute(() -> {
                try{
                    R result = converter.apply(source);
                    results.set(index, result);
                }finally {
                    latch.countDown();
                }

            });
            i ++;
        }

        try {
            latch.await(); // 等待所有任务完成
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }

        return results;
    }


}
