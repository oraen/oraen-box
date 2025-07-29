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
public class ItemInfo {

    Long itemId;

    String itemName;

    String itemDesc;

    String itemImage;

    Long itemPrice;

    Integer itemStock;

    Integer itemStatus;

    Long shopId;

    String shopName;

    List<ItemInfoTag> itemInfoTags;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ItemInfoTag{
        String title;

        String icon;

        String value;

    }
}
