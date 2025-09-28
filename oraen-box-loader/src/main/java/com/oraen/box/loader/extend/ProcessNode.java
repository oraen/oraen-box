package com.oraen.box.loader.extend;

import com.oraen.box.loader.DataLoader;
import com.oraen.box.loader.LoadContext;
import com.oraen.box.loader.RetryCommand;

@SuppressWarnings("unchecked")
public interface ProcessNode<P, R> extends DataLoader<Object> {

    @Override
    default Object getData(LoadContext context) {
        P initParam = (P)context.getInitParam(Object.class);
        R resp = (R)context.getResp(Object.class);
        return process(initParam, resp, context);
    }

    @Override
    default Object fallback(LoadContext context, Throwable e){
        P initParam = (P)context.getInitParam(Object.class);
        R resp = (R)context.getResp(Object.class);
        return fallback(initParam, resp, e, context);
    }

    @Override
    default RetryCommand needRetry(LoadContext context, Throwable e) {
        P initParam = (P)context.getInitParam(Object.class);
        R resp = (R)context.getResp(Object.class);
        return needRetry(initParam, resp, context, e);
    }

    Object process(P param, R resp, LoadContext context);

    default Object fallback(P param, R resp, Throwable t, LoadContext context){
        throw t instanceof RuntimeException ? (RuntimeException)t : new RuntimeException(t);
    }

    default RetryCommand needRetry(P param, R resp, LoadContext context, Throwable e) {
        return RetryCommand.KEEP;
    }

}
