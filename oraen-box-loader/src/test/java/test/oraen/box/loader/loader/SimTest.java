package test.oraen.box.loader.loader;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.oraen.box.common.util.JSONUtil;
import com.oraen.box.common.util.ListUtil;
import com.oraen.box.loader.LoadContext;
import com.oraen.box.loader.extend.ParallelDataBuilder;
import com.oraen.box.loader.extend.ProcessNode;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SimTest {

    @Test
    public void test() throws Exception {
        ParallelDataBuilder<Param, Resp> builder = new ParallelDataBuilder<Param, Resp>()
                //添加工作节点，可以根据自己编码习惯一行加入单个或者多个
                .addNodes(new GetUserInfoNode(), new GetMapInfoNode())
                .addNodes(new GetUserOrderNode())
                //设置超时时间
                .setExecTimeout(1000L)
                //确保节点之间没出现循环依赖，
                .ensure();

        Param initParam = Param.builder()
                .token("asdasdasdasd")
                .lat("18.444369")
                .lng("-97.3794933")
                .build();

        Resp resp = Resp.builder()
                .lat(initParam.lat)
                .lng(initParam.lng)
                .build();

        builder.buildResp(initParam, resp);
        System.out.println(JSONUtil.toJson(resp));

    }




}

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
class Param{
    String token;
    String lat;
    String lng;
}

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
class Resp{
    String cityId;
    String userName;
    Long userId;
    String lat;
    String lng;
    List<OrderDetail> orderList;


    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class OrderDetail{
        Long orderId;
        Long orderCreateTime;
    }
}

class GetUserInfoNode implements ProcessNode<Param, Resp> {

    @Override
    public Object process(Param param, Resp resp, LoadContext context) {
        String token = param.getToken();
        //mock解析token操作

        //直接给最终要返回的结果设置值
        resp.setUserId(20L);
        resp.setUserName("corki");

        //builder模式下一般用不到节点的返回结果，可以返回null
        return null;
    }

    @Override
    public String name() {
        return "getUserInfo";
    }

    @Override
    public List<String> dependencies() {
        return Collections.emptyList();
    }
}

class GetMapInfoNode implements ProcessNode<Param, Resp> {

    @Override
    public Object process(Param param, Resp resp, LoadContext context) {
        String lat = param.getLat();
        String lng = param.getLng();
        //mock解析经纬度...

        resp.setCityId("211");
        return null;
    }

    @Override
    public String name() {
        return "getMapInfo";
    }

    @Override
    public List<String> dependencies() {
        return Collections.emptyList();
    }
}


class GetUserOrderNode implements ProcessNode<Param, Resp> {

    @Override
    public Object process(Param param, Resp resp, LoadContext context) {
        //mock获取用户的所有订单
        List<Resp.OrderDetail> orderDetails = new ArrayList<>();
        for(int i = 0; i < 4; i ++){
            orderDetails.add(Resp.OrderDetail.builder()
                            .orderId(i + 1000L)
                    .build());
        }

        resp.setOrderList(orderDetails);

        //异步获取各订单的详情，不堵塞当前节点，但是需全部完成后主流程才能完成
        for(Resp.OrderDetail orderDetail : orderDetails){
            //调用context的submitTask的方法用于提交任务，true标识堵塞主流程
            context.submitTask(() -> {
                orderDetail.setOrderCreateTime(1000000 + orderDetail.getOrderId());
            }, true);
        }
        return null;
    }

    @Override
    public String name() {
        return "getUserOrder";
    }

    @Override
    public List<String> dependencies() {
        return ListUtil.of("getUserInfo");
    }
}

