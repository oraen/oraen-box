package test.oraen.box.loader.loader;

import oraen.box.common.util.ListUtil;
import oraen.box.common.util.ThreadUtil;
import oraen.box.loader.ExecLog;
import oraen.box.loader.LoadContext;
import oraen.box.loader.ProcessNode;
import oraen.box.loader.core.CommonLoaderHandler;
import oraen.box.loader.core.CommonMapDataLoaderContainer;
import oraen.box.loader.core.LoadUtil;
import org.apache.commons.lang3.RandomUtils;
import org.junit.jupiter.api.Test;
import test.oraen.box.loader.entry.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class TestLoadersTest {


    @Test
    public void testHandler() {

        MainLoader mainLoader = new MainLoader();
        CommonMapDataLoaderContainer commonMapDataLoaderContainer = new CommonMapDataLoaderContainer()
                .addDataLoader(mainLoader)
                .addDataLoader(new GetUserInfo())
                .addDataLoader(new GetMapInfo())
                .addDataLoader(new GetControlTab())
                .addDataLoader(new GetCouponInfo())
                .addDataLoader(new GetItemList())
                .addDataLoader(new GetItemInfo())
                .addDataLoader(new GetItemTag())
                .addDataLoader(new GetOrderInfo())
                .addDataLoader(new GetRecommendInfo())
                .addDataLoader(new GetSearchInfo())
                .addDataLoader(new GetShopInfo());


        ArrayList<Executor> defaultExecutors = new ArrayList<>();
        int executorCount = 4; // 默认线程池数量
        // 获取 CPU 核心数
        int cpuCores = Runtime.getRuntime().availableProcessors();
        int corePoolSize = cpuCores * 4;
        int maximumPoolSize = cpuCores * 4;

        for(int i = 0; i < executorCount; i++) {
            ThreadPoolExecutor executor = new ThreadPoolExecutor(
                    corePoolSize,
                    maximumPoolSize,
                    120L,
                    TimeUnit.SECONDS,
                    new LinkedBlockingQueue<>(),
                    Executors.defaultThreadFactory(),
                    new ThreadPoolExecutor.AbortPolicy()
            );
            defaultExecutors.add(executor);
        }

        CommonLoaderHandler commonLoaderHandler = new CommonLoaderHandler(commonMapDataLoaderContainer, defaultExecutors, 2000);

        MainPage initResp = new MainPage();
        ExecLog execLog = commonLoaderHandler.execDataLoadWithLog(mainLoader, MainParam.builder().lat("123.12").lng("32.s").appType(1).build(), initResp);
        if(initResp.getItemInfos() != null) {
            System.out.println("viiv");
        }
       // System.out.println(execLog.showExeLog());
    }

    public static class MainLoader implements ProcessNode<MainParam, MainPage>{

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            return null;
        }

        @Override
        public String name() {
            return "main";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("controlTab", "couponInfo", "itemList", "mapInfo", "orderInfo", "recommendInfo", "searchInfo", "shopList", "userInfo", "itemTag", "itemInfo");
        }
    }

    public static class GetControlTab implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            List<ControlTab> controlTabs = new ArrayList<>();
            if(resp.getUserInfo().getUserId() % 2 == 0) {
                controlTabs.add(ControlTab.builder()
                        .word("Control Tab 1")
                        .icon("Control Tab Content 1")
                                .type(1)
                        .build());
                controlTabs.add(ControlTab.builder()
                        .word("Control Tab 2")
                        .icon("Control Tab Content 2")
                                .type(1)
                        .build());
                resp.setControlTabs(controlTabs);
            } else {
                controlTabs.add(ControlTab.builder()
                        .word("Control Tab 3")
                        .icon("Control Tab Content 3")
                        .type(1)
                        .build());
                controlTabs.add(ControlTab.builder()
                        .word("Control Tab 4")
                        .icon("Control Tab Content 4")
                        .type(1)
                        .build());
                controlTabs.add(ControlTab.builder()
                        .word("Control Tab 3")
                        .icon("Control Tab Content 3")
                        .type(1)
                        .build());
                controlTabs.add(ControlTab.builder()
                        .word("Control Tab 5")
                        .icon("Control Tab Content 5")
                        .type(2)
                        .build());
            }

            ThreadUtil.sleep(1000);

            resp.setControlTabs(controlTabs);
            return controlTabs;
        }

        @Override
        public String name() {
            return "controlTab";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("userInfo");
        }
    }

    public static class GetCouponInfo implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            if(context.getDataLoadData("userInfo", Object.class) == null) {
                System.out.println("User info and couponInfo not loaded, skipping coupon info loading...");
                return Collections.emptyList();
            }

            List<CouponInfo> couponInfos = new ArrayList<>();
            couponInfos.add(CouponInfo.builder()
                    .couponId(1L)
                    .title("Coupon 1")
                    .description("Coupon Description 1")
                    .icon("coupon_icon_1.png")
                    .type(2)
                    .build());
            couponInfos.add(CouponInfo.builder()
                    .couponId(2L)
                    .title("Coupon 2")
                    .description("Coupon Description 2")
                    .icon("coupon_icon_2.png")
                    .type(2)
                    .build());

            ThreadUtil.sleep(500);

            resp.setCouponInfos(couponInfos);
            return couponInfos;
        }

        @Override
        public String name() {
            return "couponInfo";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("userInfo");
        }
    }

    public static class GetItemList implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {

            List<ItemInfo> itemInfos = new ArrayList<>();
            itemInfos.add(ItemInfo.builder()
                    .itemId(1L)
                    .itemName("item 1")
                    .itemDesc("item Description 1")
                    .build());

            itemInfos.add(ItemInfo.builder()
                    .itemId(10L)
                    .itemName("item 10")
                    .itemDesc("item Description 1")
                    .build());

            itemInfos.add(ItemInfo.builder()
                    .itemId(11L)
                    .itemName("item 11")
                    .itemDesc("item Description 11")
                    .build());

            itemInfos.add(ItemInfo.builder()
                    .itemId(23L)
                    .itemName("item 23")
                    .itemDesc("item Description 23")
                    .build());

            itemInfos.add(ItemInfo.builder()
                    .itemId(24L)
                    .itemName("item 24")
                    .itemDesc("item Description 24")
                    .build());

            ThreadUtil.sleep(250);

            resp.setItemInfos(itemInfos);
            return itemInfos;
        }

        @Override
        public String name() {
            return "itemList";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("mapInfo", "userInfo");
        }
    }

    public static class GetItemInfo implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            resp.getItemInfos().forEach(itemInfo -> {

                context.submitTask(() -> {
                    ThreadUtil.sleep(200);
                    itemInfo.setItemImage("item_image_" + itemInfo.getItemId() + ".png");
                    itemInfo.setItemPrice(100L * itemInfo.getItemId());
                    itemInfo.setItemStock(10);
                    itemInfo.setItemStatus((int) (itemInfo.getItemId() % 4));
                    itemInfo.setShopId(RandomUtils.nextLong(1, 1000));
                    itemInfo.setShopName("Shop " + itemInfo.getItemId());
                });

            });

            return null;
        }

        @Override
        public String name() {
            return "itemInfo";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("itemList");
        }
    }

    public static class GetItemTag implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            List<ItemInfo> itemInfos = resp.getItemInfos();
            LoadUtil.parallelExec(itemInfos, itemInfo -> {
                ThreadUtil.sleep(1000);

                if(param.getAppType() == 1){
                    List<ItemInfo.ItemInfoTag> itemInfoTags = new ArrayList<>();
                    itemInfoTags.add(ItemInfo.ItemInfoTag.builder()
                            .title("Tag 1")
                            .icon("tag_icon_1.png")
                            .value("Value 1")
                            .build());
                    itemInfoTags.add(ItemInfo.ItemInfoTag.builder()
                            .title("Tag 2")
                            .icon("tag_icon_2.png")
                            .value("Value 2")
                            .build());
                    itemInfo.setItemInfoTags(itemInfoTags);
                }else{
                    List<ItemInfo.ItemInfoTag> itemInfoTags = new ArrayList<>();
                    itemInfoTags.add(ItemInfo.ItemInfoTag.builder()
                            .title("Tag 3")
                            .icon("tag_icon_3.png")
                            .value("Value 3")
                            .build());
                    itemInfoTags.add(ItemInfo.ItemInfoTag.builder()
                            .title("Tag 4")
                            .icon("tag_icon_4.png")
                            .value("Value 4")
                            .build());
                    itemInfo.setItemInfoTags(itemInfoTags);
                }

            }, context);

            return null;
        }

        @Override
        public String name() {
            return "itemTag";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("itemList");
        }
    }

    public static class GetMapInfo implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            MapInfo.MapInfoBuilder lng = MapInfo.builder()
                    .areaId(1)
                    .cityId(300212)
                    .countryId(3002)
                    .lat(param.getLat())
                    .lng(param.getLng());
            resp.setMapInfo(lng
                    .build());

            return lng;
        }

        @Override
        public String name() {
            return "mapInfo";
        }

        @Override
        public List<String> dependencies() {
            return null;
        }
    }

    public static class GetOrderInfo implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            throw new RuntimeException("获取订单异常");
        }

        @Override
        public Object fallback(MainParam param, MainPage resp, Throwable e, LoadContext context) {
            System.out.println("异常了哇，Fallback for GetOrderInfo: " + e.getMessage());
            List<OrderInfo> orderInfos = new ArrayList<>();
            orderInfos.add(OrderInfo.builder()
                    .orderId(1L)
                    .build());
            resp.setOrderInfos(orderInfos);
            return orderInfos;
        }


        @Override
        public String name() {
            return "orderInfo";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("userInfo");
        }
    }

    public static class GetRecommendInfo implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {
            List<RecommendInfo.RecommendItem> collect = resp.getCouponInfos().stream().map(couponInfo -> RecommendInfo.RecommendItem.builder()
                    .title("recommend " + couponInfo.getCouponId())
                    .tip(couponInfo.getTitle())
                    .imageUrl(couponInfo.getIcon())
                    .build()).collect(Collectors.toList());

            RecommendInfo build = RecommendInfo.builder()
                    .recommendItems(collect)
                    .build();
            resp.setRecommendInfos(build);
            return build;
        }


        @Override
        public String name() {
            return "recommendInfo";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("userInfo", "mapInfo", "itemList", "couponInfo");
        }
    }


    public static class GetSearchInfo implements ProcessNode<MainParam, MainPage> {



        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {


            ThreadUtil.sleep(1000);


            resp.setSearchInfo(SearchInfo.builder()
                            .word(ListUtil.of("Search Word 1", "Search Word 2", "Search Word 3"))
                    .build());
            return null;
        }


        @Override
        public String name() {
            return "searchInfo";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("recommendInfo");
        }
    }

    public static class GetShopInfo implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {

            ThreadUtil.sleep(1000);
            List<ShopInfo> shopInfos = new ArrayList<>();
            shopInfos.add(ShopInfo.builder()
                    .shopId(1L)
                    .shopName("Shop 1")
                    .message("shop_image_1.png")
                    .build());

            shopInfos.add(ShopInfo.builder()
                    .shopId(2L)
                    .shopName("Shop 2")
                    .message("shop_image_2.png")
                    .build());

            resp.setShopInfos(shopInfos);
            resp.setShopInfos(shopInfos);
            return null;
        }


        @Override
        public String name() {
            return "shopList";
        }

        @Override
        public List<String> dependencies() {
            return ListUtil.of("mapInfo");
        }
    }

    public static class GetUserInfo implements ProcessNode<MainParam, MainPage> {

        @Override
        public Object process(MainParam param, MainPage resp, LoadContext context) {

            ThreadUtil.sleep(50);
            UserInfo user = UserInfo.builder()
                    .userId(12323L)
                    .userName("User ")
                    .userPhone("1234567890")
                    .build();
            resp.setUserInfo(user);
            return user;
        }


        @Override
        public String name() {
            return "userInfo";
        }

        @Override
        public List<String> dependencies() {
            return new ArrayList<>();
        }
    }









}
