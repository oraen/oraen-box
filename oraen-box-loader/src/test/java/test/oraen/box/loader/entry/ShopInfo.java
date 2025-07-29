package test.oraen.box.loader.entry;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShopInfo {

    Long shopId;

    String shopName;

    String message;

    String shopAddress;

    String shopPhone;

    String shopEmail;

    List<ShopTag> tags;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ShopTag{
        String title;

        String icon;

        String value;

    }



}
