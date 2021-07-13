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

#include "QueryRunner/QueryRunner.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/CiderResultProvider.h"
#include "QueryEngine/CiderArrowResultProvider.h"

/*
 * Class:     com_mapd_CiderJNI
 * Method:    processBlocks
 * Signature: (Ljava/lang/String;Ljava/lang/String;[J[J[J[JI)I
 */
JNIEXPORT jint JNICALL Java_com_mapd_CiderJNI_processBlocks(
    JNIEnv* env, jclass cls, jstring sql, jstring schema,
    jlongArray dataValues, jlongArray dataNulls,
    jlongArray resultValues, jlongArray resultNulls, jint rowCount) {

  jsize dataValuesLen = env->GetArrayLength(dataValues);
  jsize dataNullsLen = env->GetArrayLength(dataNulls);

  const char* sqlPtr = env->GetStringUTFChars(sql, nullptr);
  const char* schemaPtr = env->GetStringUTFChars(schema, nullptr);

  jlong* dataValuesPtr = env->GetLongArrayElements(dataValues, 0);
  jlong* dataNullsPtr = env->GetLongArrayElements(dataNulls, 0);

  printf("processing within JNI...\n");
  std::vector<int8_t*> dataBuffers;
  for (int i = 0; i < dataValuesLen; i++) {
    dataBuffers.push_back((int8_t *)(dataValuesPtr[i]));
  }

  auto dp = std::make_shared<BufferCiderDataProvider>(dataValuesLen, 0, dataBuffers, rowCount);
  auto rp = std::make_shared<CiderArrowResultProvider>();
  auto res_itr = QueryRunner::QueryRunner::get()->ciderExecute(
      sqlPtr,
      ExecutorDeviceType::CPU,
      /*hoist_literals=*/true,
      /*allow_loop_joins=*/false,
      /*just_explain=*/false,
      dp,
      rp);
  auto res = res_itr->next(rowCount);
  auto crt_row = res->getRows()->getNextRow(true, true);
  std::shared_ptr<arrow::RecordBatch> record_batch =
      std::any_cast<std::shared_ptr<arrow::RecordBatch>>(rp->convert());

  env->ReleaseStringUTFChars(sql, sqlPtr);
  env->ReleaseStringUTFChars(schema, schemaPtr);
  return crt_row.size();
}
