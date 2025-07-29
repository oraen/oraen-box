package test.oraen.box.loader.entry;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MapInfo {

    Integer areaId;

    Integer cityId;

    Integer countryId;

    String lat;

    String lng;
}
