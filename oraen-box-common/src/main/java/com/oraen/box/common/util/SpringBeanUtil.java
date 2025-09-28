package com.oraen.box.common.util;

import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Controller;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Component
public class SpringBeanUtil implements ApplicationContextAware {

    private static ApplicationContext context;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) {
        SpringBeanUtil.context = applicationContext;
    }

    public static <T> T getBean(Class<T> clazz) {
        if (context == null) {
            throw new IllegalStateException("ApplicationContext not initialized");
        }
        return context.getBean(clazz);
    }

    public static <T> T getBean(String name, Class<T> clazz) {
        if (context == null) {
            throw new IllegalStateException("ApplicationContext not initialized");
        }
        return context.getBean(name, clazz);
    }

    /**
     * 获取指定类型的所有 Bean，返回 Map，key 是 Bean 名称，value 是 Bean 实例
     */
    public static <T> Map<String, T> getBeanMap(Class<T> clazz) {
        if (context == null) {
            throw new IllegalStateException("ApplicationContext not initialized");
        }
        return context.getBeansOfType(clazz);
    }


    public static <T> List<T> getBeanList(Class<T> clazz) {
        if (context == null) {
            throw new IllegalStateException("ApplicationContext not initialized");
        }
        return new ArrayList<>(context.getBeansOfType(clazz).values());
    }

    public static String getBeanName(Object obj){
        return getBeanMap(obj.getClass()).entrySet().stream()
                .filter(entry -> entry.getValue().equals(obj))
                .map(Map.Entry::getKey)
                .findFirst()
                .orElseGet(() -> getBeanNameOld(obj));
    }

    public static String getBeanNameOld(Object obj){

        Class<?> clazz = obj.getClass();
        Component componentAnnotation = clazz.getAnnotation(Component.class);
        if(componentAnnotation != null && !componentAnnotation.value().isEmpty()) {
            return componentAnnotation.value();
        }

        Service serviceAnnotation = clazz.getAnnotation(Service.class);
        if(serviceAnnotation != null && !serviceAnnotation.value().isEmpty()) {
            return serviceAnnotation.value();
        }

        Repository repository = clazz.getAnnotation(Repository.class);
        if (repository != null && !repository.value().isEmpty()) {
            return repository.value();
        }

        Controller controller = clazz.getAnnotation(Controller.class);
        if (controller != null && !controller.value().isEmpty()) {
            return controller.value();
        }

        Configuration configuration = clazz.getAnnotation(Configuration.class);
        if (configuration != null && !configuration.value().isEmpty()) {
            return configuration.value();
        }

        return ReflectUtil.getSimClassName(obj).toLowerCase();
    }


}