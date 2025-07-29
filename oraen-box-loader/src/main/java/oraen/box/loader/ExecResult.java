package oraen.box.loader;

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


    private Object result;
    private Throwable exception;
    private boolean isSuccess;
    private boolean useFallback;
    private boolean isCompleted;
    private long execTime = -1;
    private volatile int status;

    public<T> T getResult() {
        return (T) result;
    }

}