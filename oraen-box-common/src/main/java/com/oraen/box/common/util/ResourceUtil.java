package com.oraen.box.common.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Objects;

public final class ResourceUtil {

    private ResourceUtil() {
    }

    /**
     * 获取 resource 下文件的 URL
     *
     * @param resourcePath 相对于 resources 的路径，如：
     *                     "corpus/corpusTest.txt"
     * @return URL（jar:file:/... 或 file:/...）
     */
    public static URL getResourceUrl(String resourcePath) {
        Objects.requireNonNull(resourcePath, "resourcePath must not be null");

        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        URL url = classLoader.getResource(resourcePath);

        if (url == null) {
            throw new IllegalArgumentException("Resource not found: " + resourcePath);
        }
        return url;
    }

    public static String readResourceAsString(String resourcePath, Charset charset) {
        URL url = getResourceUrl(resourcePath);

        try (InputStream is = url.openStream();
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {

            byte[] buffer = new byte[8192];
            int len;
            while ((len = is.read(buffer)) != -1) {
                baos.write(buffer, 0, len);
            }
            return new String(baos.toByteArray(), charset);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}