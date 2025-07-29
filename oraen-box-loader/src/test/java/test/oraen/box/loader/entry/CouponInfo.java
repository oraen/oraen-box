package test.oraen.box.loader.entry;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CouponInfo {

    Long couponId;

    String title;

    String description;

    String icon;

    Integer type;

}
