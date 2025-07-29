package oraen.box.loader.core;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Getter
public class DataLoadTask {

    //这个任务依赖的任务，
//    @Getter
//    private final Set<DataLoadTask> waitingForTasks = ConcurrentHashMap.newKeySet();

    //这个任务依赖的任务数
    private final AtomicInteger waitingForTasksNum = new AtomicInteger(0);

    //记录依赖这个任务的任务，不可并发写入
    private final List<DataLoadTask> waitingTasks = new ArrayList<>();

    @Setter
    private Runnable executor;

    @Setter
    private final String name;

    public DataLoadTask(String name) {
        this.name = name;
    }

    public void addDependency(DataLoadTask task) {
//        waitingForTasks.add(task);
        waitingForTasksNum.incrementAndGet();
        task.waitingTasks.add(this);
    }

    public boolean removeDependencyAndIsReady() {
        int v = waitingForTasksNum.decrementAndGet();
        return v == 0;
    }

    //不可在并发时单独调用，可能有并发问题
    public boolean isReady() {
        return waitingForTasksNum.get() == 0;
    }

}
