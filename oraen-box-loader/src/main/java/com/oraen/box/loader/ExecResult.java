package com.oraen.box.loader;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@SuppressWarnings("unchecked")
public class ExecResult {

    public static final int STATUS_INIT = 0;
    public static final int STATUS_WAITING = 10;
    public static final int STATUS_EXECUTING = 20;
    public static final int STATUS_SUCCESS = 30;
    public static final int STATUS_FALLBACK = 40;
    public static final int STATUS_ERROR = 50;
    public static final int STATUS_ABANDON = 60;


    //返回结果
    private Object result;
    //异常
    private Throwable exception;
    //是否成功，执行了Fallback不算成功
    private boolean isSuccess;
    //是否执行了回调
    private boolean useFallback;
    //是否已经完成，执行了Fallback也算完成
    private boolean isCompleted;
    //执行时间
    private long execTime = -1;
    //执行状态
    private int status;
    //重试次数,没重试为1
    private int retry = 1;

    public<T> T getResult() {
        return (T) result;
    }

}