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
public class UserInfo {

    Long userId;

    String userName;

    String userPhone;

    String userEmail;

    Integer status;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MemberInfo {
        String memberId;
        String memberName;
        Integer level;
        LocalDateTime expireTime;
    }

}
