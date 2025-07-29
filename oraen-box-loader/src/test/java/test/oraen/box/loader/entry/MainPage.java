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
public class MainPage {

    List<ControlTab> controlTabs;

    List<ItemInfo> itemInfos;

    MapInfo mapInfo;

    List<OrderInfo> orderInfos;

    RecommendInfo recommendInfos;

    SearchInfo searchInfo;

    List<ShopInfo> shopInfos;

    UserInfo userInfo;

    List<CouponInfo> couponInfos;

}
