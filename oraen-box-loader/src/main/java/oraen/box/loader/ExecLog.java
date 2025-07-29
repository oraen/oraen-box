package oraen.box.loader;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import oraen.box.common.util.JSONUtil;

import java.util.List;
import java.util.Map;
import java.util.Objects;

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
