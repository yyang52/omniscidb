/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include <gtest/gtest.h>
#include <iostream>
#include "RelAlgDagBuilder.cpp"

using namespace std;

TEST_F(TestOModuler, SimpleProject) {
  cout << "hello world" << endl;
  OModuler oModuler();
  std::shared_ptr<RelAlgNode> raPtr = oModuler.ast_convert(nullptr);

  auto workUnit =
      OMModules::kernel.createWorkUnit(node, context);  // will do code gen internally

  while (hasMoreData()) {
    inputData = OMModules::Data.convert(bufferFromDS, context) outputData =
        workUnit.executeWithData(inputData, context)
  }
}
