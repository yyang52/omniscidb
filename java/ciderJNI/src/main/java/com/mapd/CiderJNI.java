package com.mapd;

import java.io.IOException;

public class CiderJNI {

    static {
      System.loadLibrary("ciderjni");
    }

    public static native int processBlocks(String sql, String schema,
            long[] dataBuffers, long[] dataNulls,
            long[] resultBuffers, long[] resultNulls,
            int rowCount);
}
