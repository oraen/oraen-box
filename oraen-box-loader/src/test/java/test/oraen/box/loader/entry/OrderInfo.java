package test.oraen.box.loader.entry;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OrderInfo {
    Long orderId;
    Integer status;
    Long shopId;
    Long userId;
    Long money;
    LocalDateTime createTime;
    LocalDateTime updateTime;
    LocalDateTime payTime;
    LocalDateTime finishTime;
}