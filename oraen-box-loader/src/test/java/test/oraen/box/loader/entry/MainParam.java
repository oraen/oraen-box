package test.oraen.box.loader.entry;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MainParam {

    String token;

    Integer appType;

    String lat;

    String lng;
}
