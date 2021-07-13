package com.mapd;

import java.io.IOException;

public class CiderJNI {

    static {
      System.loadLibrary("ciderjni");
    }

    public native void sayHello();

    public static void main(String[] args) {
      new CiderJNI().sayHello();
    }
}
