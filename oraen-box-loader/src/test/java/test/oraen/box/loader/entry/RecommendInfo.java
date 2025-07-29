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
public class RecommendInfo {
    List<RecommendItem> recommendItems;


    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RecommendItem{
        String title;

        String tip;

        String imageUrl;
    }
}
