package oraen.box.loader.spring;

import oraen.box.common.util.SpringBeanUtil;
import oraen.box.loader.DataLoader;


public abstract class CommonSpringDataLoader<T> implements DataLoader<T> {

    private volatile String name = null;

    @Override
    public String name() {
        if(name == null) {
            synchronized (this) {
                if(name == null) {
                    name = SpringBeanUtil.getBeanName(this);
                }
            }
        }

        return name;
    }
}
