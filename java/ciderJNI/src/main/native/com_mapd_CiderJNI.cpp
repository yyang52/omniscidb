/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include "com_mapd_CiderJNI.h"

JNIEXPORT void JNICALL Java_com_mapd_CiderJNI_sayHello(JNIEnv*, jobject) {
  printf("Hello World\n");
#ifdef __cplusplus
  printf("__cplusplus is defined\n");
#else
  printf("__cplusplus is NOT defined\n");
#endif
  return;
}