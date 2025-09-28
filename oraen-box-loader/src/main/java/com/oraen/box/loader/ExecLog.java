package com.oraen.box.loader;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.oraen.box.common.util.JSONUtil;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@SuppressWarnings("unchecked")
public class ExecLog {

    private Object data;

    private LoadContext context;

    public<T> T getData() {
        return (T) data;
    }

    public String showExeLog(){
        return JSONUtil.toJson(this);
    }

}
